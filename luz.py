import argparse
import PIL.Image
import random
import time
import json
import socket

import numpy as np
import pandas as pd

from mpi4py import MPI

from vectors.Ray import Ray
from vectors.Point import Point

from geometry import Object
from geometry.Sphere import Sphere
from geometry.Sphere import Skybox
from geometry import Plane
from geometry import Triangle

ASCII_CHARS = ["@", "#", "$", "%", "?", "*", "+", ";", ":", ",", "."]

class Material:
    def __init__(self, name, diffuse = 0.5, reflection = 0.5, shiny = 0.5, k = 8,
                 color = [255, 255, 255], texture = None):

        self.name = name
        self.color = np.array(color)
        self.diffuse = diffuse
        self.reflection = reflection
        self.shiny = shiny
        self.k = k
        self.texture = None

        if(texture is not None):
            self.texture = Image.open(texture)

class Light(Sphere):
    """A light is a sphere that is a bounce terminator"""
    def __init__(self, origin, radius, brightness, color):
        self.origin = np.array(origin)
        self.radius = radius
        self.brightness = brightness
        self.color = np.array(color) / 255.0

    # a light can never shadow another light
    def collides_with(self, ray):
        return(False, None)

    # a light intersection is basically a sphere intersection, but just
    # returns the color of the light
    def intersect(self, ray, scene, bounce, max_bounces):
        origin, distance = self.geo_intersect(ray)

        if(origin is None):
            return(scene.background, None, 1)

        return(self.color * 255.0, distance, 1)


# based on https://www.scratchapixel.com/lessons/mathematics-physics-for-computer-graphics/lookat-function/framing-lookat-function
# and https://www.youtube.com/watch?v=LRN_ewuN_k4
class Camera:

    def __init__(self, origin, target, aperture = 0.1, length = 1.0, samples = 50):
        self.origin = np.array(origin)
        self.target = np.array(target)
        self.aperture = aperture
        self.length = length
        self.samples = samples

        forward = Point(self.target - self.origin, None)
        forward.normalize()

        right = Point(np.cross(forward.coords, np.array([0, 1, 0])), None)
        right.normalize()
        up = Point(np.cross(right.coords, forward.coords), None)
        up.normalize()

        self.rotation_matrix = np.array([
            right.coords, up.coords, forward.coords
        ])

        # distance to focal point
        focus_dir = self.target - self.origin
        self.focus = np.sqrt(sum( focus_dir * focus_dir ))

        # camera circular bokeh
        # float angle = RandomFloat01(state) * 2.0f * c_pi;
        # float radius = sqrt(RandomFloat01(state));
        # float2 offset = float2(cos(angle), sin(angle)) * radius * ApertureRadius;

        if(self.aperture == 0.0):
            self.samples = 1

class Scene:
    def __init__(self, filename = "scene.json"):

        self.lights = []
        self.objects = []

        # read in the json
        with open(filename, "r") as json_file:
            scene = json.load(json_file)

        for light in scene['scene']['lights']:
            l = Light(origin = light['origin'], radius = light['radius'],
                                     brightness = light['brightness'], color = light['color'])
            self.lights.append(l)
            self.objects.append(l)

        for object in scene['scene']['objects']:

            texture = None
            scale = 1.0

            if 'texture' in object:
                texture = object['texture']

            if 'scale' in object:
                scale = object['scale']

            if(object['shape'] == "Sphere"):

                self.objects.append(Sphere(origin = object['origin'], radius = object['radius'],
                                           diffuse = object['diffuse'], reflection = object['reflection'],
                                           shiny = object['shiny'], k = object['k'],
                                           refraction = object['refraction'], index = object['index'],
                                           color = object['color'],
                                           texture = texture, uv = object['uv'] ))
            elif(object['shape'] == "Skybox"):

                self.objects.append(Skybox(origin = object['origin'], radius = object['radius'],
                                           diffuse = object['diffuse'], reflection = object['reflection'],
                                           shiny = object['shiny'], k = object['k'],
                                           refraction = object['refraction'], index = object['index'],
                                           color = object['color'],
                                           texture = texture, uv = object['uv'] ))

            elif(object['shape'] == "Plane"):

                self.objects.append(Plane.Plane(origin = object['origin'], normal = object['normal'],
                                          diffuse = object['diffuse'], reflection = object['reflection'],
                                          refraction = object['refraction'], index = object['index'],
                                          shiny = object['shiny'], k = object['k'],
                                          color = object['color'],
                                          texture = texture, scale = scale))
            elif(object['shape'] == "Triangle"):
                self.objects.append(Triangle.Triangle(v0 = object['v0'], v1 = object['v1'], v2 = object['v2'],
                                          diffuse = object['diffuse'], reflection = object['reflection'],
                                          refraction = object['refraction'], index = object['index'],
                                          shiny = object['shiny'], k = object['k'],
                                          color0 = object['color0'],
                                          color1 = object['color1'],
                                          color2 = object['color2'] ))

        camera = scene['scene']['camera']
        self.camera = Camera(origin = camera['origin'], target = camera['target'],
                             length = camera['length'], aperture = camera['aperture'],
                             samples = camera['samples'])

        self.ambient = np.array(scene['scene']['ambient'])
        self.background = np.array(scene['scene']['background'])


    def render(self, size, rank, img_height, img_width, max_bounces):

      worker_pixels = np.zeros((img_height, img_width, 3))
      total_rays = 1

      # displacement from the top left corner to the middle of the pixel
      PIXEL_WIDTH = (2.0 / (img_width + 1))
      PIXEL_HEIGHT = (2.0 / (img_height + 1))

      stats = {'rank': rank, 'hostname': socket.gethostname(), 'pixels': 0, 'geo_intersections': 0, 'time': time.time()}

      for y in range(0, img_height):
          for x in range(0, img_width):

              if(total_rays % (size - 1) == (rank - 1)):

                  stats['pixels'] = stats['pixels'] + 1

                  # create a ray from the default camera to this pixel
                  # camera location, (image plane x, image plane y, image plane z)
                  plane = np.array([((x + 1) * PIXEL_WIDTH) - 1, ((img_height - y + 1) * PIXEL_HEIGHT) - 1, self.camera.length])
                  ray = Ray(self.camera.origin, plane)

                  # now we apply the camera rotation to the ray direction
                  #ray.direction = np.matmul(self.camera.rotation_matrix, ray.direction)
                  ray.direction = np.matmul(ray.direction, self.camera.rotation_matrix)

                  focal_point = self.camera.origin + (ray.direction * self.camera.focus)
                  image_plane = self.camera.origin + (ray.direction * self.camera.length)

                  # might be able to cache these offsets in the camera class to
                  # avoid recomputing for every primary ray
                  # todo: circular vs square bokeh
                  if(self.camera.samples > 1):
                      sample_origin = [image_plane + np.matmul(np.append( (np.random.rand(2) - 0.5) * self.camera.aperture, 0), self.camera.rotation_matrix) for sample in range(0, self.camera.samples)]
                  else:
                      sample_origin = [self.camera.origin, self.camera.origin]

                  # ray.normalize()

                  for sample in range(0, self.camera.samples):

                      # calculate a new origin with random offset
                      sample_ray = Ray(sample_origin[sample], focal_point - sample_origin[sample])
                      # sample_ray.normalize()

                      zbuffer = None

                      for o in self.objects:
                          # pixel, depth, distance, geo_intersections = o.intersect(ray, self, 0, max_bounces)
                          pixel, distance, geo_intersections = o.intersect(sample_ray, self, 0, max_bounces)
                          stats['geo_intersections'] = stats['geo_intersections'] + geo_intersections
                          # if(zbuffer[y][x] is None or (distance is not None and distance < zbuffer[y][x])):
                          if(zbuffer is None or (distance is not None and distance < zbuffer)):
                              worker_pixels[y][x] += pixel
                              # zbuffer[y][x] = distance
                              zbuffer = distance

              total_rays += 1

      stats['time'] = time.time() - stats['time']
      return(worker_pixels, stats)

def resize(image, new_width = 80):
  width, height = image.size
  new_height = int(new_width * height / width)
  return image.resize((new_width, new_height))

def to_greyscale(image):
  return image.convert("L")

def pixel_to_ascii(image):
  pixels = image.getdata()
  ascii_str = "";
  for pixel in pixels:
    ascii_str += ASCII_CHARS[pixel//25];
  return ascii_str

def color_esc(pixel):
    return('\x1b[38;2;%s;%s;%sm%s\x1b[40m' % (pixel[0][0], pixel[0][1], pixel[0][2], pixel[1]))

def print_ascii(input, term_width = 80):
    image = resize(input, term_width)

    grey = to_greyscale(image)
    shapes = pixel_to_ascii(grey)
    colors = image.getdata()

    color_ascii = zip(colors, shapes)

    output = list(map(color_esc, color_ascii))

    for cursor in range(0, len(output)):
        print(output[cursor], end = '')
        if(cursor % term_width == 0):
            print("")
    print("")


def main(img_height = 200, img_width = 200, cols = 80, output_filename = 'output.png',
         max_bounces = 3, show_stats = False, long_stats = False, sd = 2.0, input_filename = 'scenes/scene.json',
         camera = None, target = None, focal = None, aperture = None, samples = None, profile = False, quiet = False):

  if(profile):
      import cProfile, pstats
      profiler = cProfile.Profile()
      profiler.enable()

  comm = MPI.COMM_WORLD
  rank = comm.Get_rank()
  size = comm.Get_size()

  scene = Scene(input_filename)

  if(camera is not None and target is not None):
      # override scene camera from command line args
      scene.camera = Camera(origin = np.array(camera, dtype=np.float32),
                       target = np.array(target, dtype=np.float32))

  if(samples is not None):
      scene.camera.samples = samples

  if(focal is not None):
      scene.camera.length = focal

  if(aperture is not None):
      scene.camera.aperture = aperture
      if aperture == 0.0:
          scene.camera.samples = 1

  if(rank != 0):
      pixels, stats = scene.render(size, rank, img_height, img_width, max_bounces)
      comm.Send(pixels, dest = 0)
      comm.send(stats, dest = 0)

  if(rank == 0):

      print("Camera focus distance: %f. Aperture: %f. Samples %i." % (scene.camera.focus, scene.camera.aperture, scene.camera.samples))
      print("Camera location: ", scene.camera.origin, " Target: ", scene.camera.target)
      start = time.time()
      pixels = np.zeros((img_height, img_width, 3))
      stats = []

      for worker in range(1, size):
          worker_pixels = np.zeros((img_height, img_width, 3))
          # worker_pixels = np.zeros( ( int((img_width * img_height) / (size - 1)) + 1, 3))
          comm.Recv(worker_pixels, source = worker)

          worker_stats = comm.recv(source = worker)

          rays = 0
          # total_rays = 1

          # for y in range(0, img_height):
          #     for x in range(0, img_width):
          #         if(total_rays % (size - 1) == (worker - 1)):
          #             pixels[y][x] = worker_pixels[rays]
          #             rays += 1
          #         # total_rays += 1

          # print("Received %i rays from worker %i." % (rays, worker))
          pixels += worker_pixels

          stats.append(worker_stats)

      pixels /= scene.camera.samples
      output = PIL.Image.fromarray(pixels.clip(0,255).astype('uint8'), 'RGB')
      output.save(output_filename)

      if(not quiet):
          print_ascii(output, 80)
          print("\033[0mThere... are... %i... lights!" %(len(scene.lights)))
      print("Finished in %f seconds. Output: %s" % (time.time() - start, output_filename))

      if(show_stats):
          print("Worker stats distribution.")

          stats_df = pd.DataFrame(stats)
          time_mean = float(stats_df[['time']].mean())
          time_sd = float(stats_df[['time']].std())

          pixel_mean = float(stats_df[['pixels']].mean())
          pixel_sd = float(stats_df[['pixels']].std())

          intersections_mean = float(stats_df[['geo_intersections']].mean())
          intersections_sd = float(stats_df[['geo_intersections']].std())

          clean = True
          for worker in stats:
              # fixme, do this as a table and loop over
              if(worker['time'] > time_mean + (sd * time_sd) or worker['time'] < time_mean - (sd * time_sd)):
                  clean = False
                  print(" - Worker %i elapsed time is more than %f sd away (diff: %f) from mean of %f." % (worker['rank'], sd, (worker['time'] - time_mean), time_mean))
              if(worker['pixels'] > pixel_mean + (sd * pixel_sd) or worker['pixels'] < pixel_mean - (sd * pixel_sd)):
                  clean = False
                  print(" - Worker %i pixel count is more than %f sd away (diff: %f) from mean of %f." % (worker['rank'], sd, (worker['pixels'] - pixel_mean), pixel_mean))
              if(worker['geo_intersections'] > intersections_mean + (sd * intersections_sd) or worker['geo_intersections'] < intersections_mean - (sd * intersections_sd)):
                  clean = False
                  print(" - Worker %i calculated intersections is more than %f sd away (diff: %f) from mean of %f." % (worker['rank'], sd, (worker['geo_intersections'] - intersections_mean), intersections_mean))
          if(clean):
              print(" - All nodes appear balanced.")

          print(" - Averages: time %f +- %f, pixels %f +- %f, intersections %f +- %f" % (
              time_mean, time_sd,
              pixel_mean, pixel_sd,
              intersections_mean, intersections_sd
          ))

          if(long_stats):
              print("Worker details:")
              for worker in stats:
                  print(" - Worker %i (%s), time: %f, pixels %i, intersections: %i." %(
                      worker['rank'],
                      worker['hostname'],
                      worker['time'],
                      worker['pixels'],
                      worker['geo_intersections']
                  ))

  if(profile):
      profiler.disable()
      stats = pstats.Stats(profiler).sort_stats('tottime')
      stats.dump_stats('profiles/rank-%d.txt' % (rank))

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='A simple MPI raytracer.')

    parser.add_argument('--bounces', type = int , default = 50, help='max number of bounces')
    parser.add_argument('--height', type = int , default = 200, help='output image height')
    parser.add_argument('--width', type = int , default = 200, help='output image width')
    parser.add_argument('--cols', type = int , default = 200, help='terminal columns for ascii output')
    parser.add_argument('--sd', type = float , default = 2.0, help='number of standard deviations from the mean to consider a node an outlier')
    parser.add_argument('--output', type = str , default = 'output.png', help='output image filename')
    parser.add_argument('--scene', type = str , default = 'scenes/scene.json', help='json scene file')
    parser.add_argument('--stats', action='store_true', help='show stats on computational distribution')
    parser.add_argument('--long', action='store_true', help='show long stats on computational distribution')

    parser.add_argument('--camera', nargs="+", default=None, help = "camera origin")
    parser.add_argument('--target', nargs="+", default=None, help = "camera target")
    parser.add_argument('--focal', type = float , default = None, help='camera focal length')
    parser.add_argument('--aperture', type = float , default = None, help='camera aperture')
    parser.add_argument('--samples', type = int , default = None, help='dof samples')

    parser.add_argument('--profile', action='store_true', help='runs profiler')
    parser.add_argument('--quiet', action='store_true', help='no ascii output')

    args = parser.parse_args()

    main(args.height, args.width, args.cols, args.output, args.bounces,
         args.stats, args.long, args.sd, args.scene, args.camera, args.target,
         args.focal, args.aperture, args.samples, args.profile, args.quiet)
