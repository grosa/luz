
import argparse
import PIL.Image
import random
import time
import json

import numpy as np
import pandas as pd

from mpi4py import MPI

ASCII_CHARS = ["@", "#", "$", "%", "?", "*", "+", ";", ":", ",", "."]

class Point:

    def __init__(self, coords):
        self.coords = coords

    def x(self):
        return self.coords[0]

    def y(self):
        return self.coords[1]

    def z(self):
        return self.coords[2]

    def normalize(self):
      norm = np.linalg.norm(self.coords)
      if norm != 0:
          self.coords = self.coords / norm

class Object:

    def intersect(self, ray, scene, bounce, max_bounces):

        origin, distance = self.geo_intersect(ray)
        total_geo_intersections = 1

        if(origin is None):
            return(scene.ambient, None, None, 1)

        normal = self.normal_at(origin)
        color = np.full(3, 0.0)
        reflected_color = np.full(3, 0.0)

        for l in scene.lights:

            lightray = Ray(l.origin, l.origin - origin)
            light_distance = lightray.distance()
            lightray.normalize()

            shadowray = Ray(origin, l.origin - origin)
            in_shadow, shadow_intersections = shadowray.in_shadow(self, scene)
            total_geo_intersections = total_geo_intersections + shadow_intersections

            if in_shadow == False:
                color += (self.color_at(origin) * l.color * l.brightness * self.diffuse * max(np.dot(normal, lightray.direction), 0) / light_distance).clip(0, 255)

                if(self.shiny > 0):
                    cam_ray = Ray(origin, scene.camera.origin - origin)
                    cam_ray.normalize()

                    halfway = cam_ray.direction + lightray.direction
                    color += (l.color * l.brightness * self.shiny * max(np.dot(normal, halfway), 0) ** self.k / light_distance).clip(0, 255)

        if(bounce < max_bounces):
            dot = -np.dot(normal, ray.direction)
            raydir = ray.direction + (2 * dot * normal)
            newray = Ray(origin, raydir)

            closest = 999
            for obj in scene.objects:
                if obj != self:
                    r_value, r_depth, r_distance, geo_intersections = obj.intersect(newray, scene, bounce + 1, max_bounces)
                    total_geo_intersections += geo_intersections
                    if(r_distance is not None and r_distance < closest):
                        closest = r_distance
                        reflected_color = r_value * self.reflection

        return((color + reflected_color + scene.ambient).clip(0, 255), origin[2], distance, total_geo_intersections)

class Ray:
    def __init__(self, origin, direction):
        self.origin = np.array(origin)
        self.direction = np.array(direction)

    def distance(self):
        a = self.direction[0] * self.direction[0]
        b = self.direction[1] * self.direction[1]
        c = self.direction[2] * self.direction[2]

        return(np.sqrt(a+b+c))

    def normalize(self):
      norm = np.linalg.norm(self.direction)
      if norm != 0:
          self.direction /= norm

    def in_shadow(self, this, scene):
        intersections = 0
        for obj in scene.objects:
            if obj is not this:
                intersections = intersections + 1
                if obj.collides_with(self):
                    return(True, intersections)
        return(False, intersections)

class Sphere(Object):
    def __init__(self, origin, radius = 1, diffuse = 0.5, reflection = 0.5, shiny = 0.5, k = 8, color = [255, 255, 255]):
        self.origin = np.array(origin)
        self.radius = radius
        self.color = np.array(color)
        self.diffuse = diffuse
        self.reflection = reflection
        self.shiny = shiny
        self.k = k

    def collides_with(self, ray):
        origin, distance = self.geo_intersect(ray)
        if(origin is None or distance is None):
            return(False)
        return(True)

    # adapted from http://viclw17.github.io/2018/07/16/raytracing-ray-sphere-intersection/
    def geo_intersect(self, ray):

        oc = ray.origin - self.origin
        a = np.dot(ray.direction, ray.direction)
        b = 2.0 * np.dot(oc, ray.direction)
        c = np.dot(oc, oc) - (self.radius * self.radius)
        discriminant = (b * b) - (4 * a * c)

        if(discriminant < 0.0):
            return(None, None)
        else:
            numerator = -b - np.sqrt(discriminant)
            if(numerator > 0.0):
                distance = numerator / (2.0 * a)
                origin = ray.origin + (ray.direction * distance)
                return(origin, distance)

            numerator = -b + np.sqrt(discriminant)
            if(numerator > 0.0):
                distance = numerator / (2.0 * a)
                origin = ray.origin + (ray.direction * distance)
                return(origin, distance)
            else:
                return(None, None)

    def normal_at(self, origin):
        dir = origin - self.origin
        norm = np.linalg.norm(dir)
        if(norm != 0):
            dir /= norm
        return(dir)

    def color_at(self, origin):
        return(self.color)

class Plane(Object):
    def __init__(self, origin, normal, diffuse, reflection, shiny = 0.5, k = 8, color1 = np.array([0,0,0]), color2 = np.array([255, 255, 255])):
        self.origin = np.array(origin)
        self.normal_v = np.array(normal)
        self.diffuse = diffuse
        self.reflection = reflection
        self.shiny = shiny
        self.k = k
        self.color1 = np.array(color1)
        self.color2 = np.array(color2)

    def collides_with(self, ray):
        origin, distance = self.geo_intersect(ray)
        if(origin is None or distance is None):
            return(False)
        return(True)

    def normal(self):
        return(self.normal_v)

    def normal_at(self, origin):
        return(self.normal_v)

    # simple checkerboard hack
    def color_at(self, origin):
        if((int(origin[0]) % 2 == 0) and (int(origin[2]) % 2 == 0)):
            return(self.color1)
        if((int(origin[0]) % 2 == 1) and (int(origin[2]) % 2 == 1)):
            return(self.color1)
        return(self.color2)

    # ray-plane intersection
    # adapted from https://stackoverflow.com/questions/23975555/how-to-do-ray-plane-intersection
    def geo_intersect(self, ray):
        denom = np.dot(self.normal(), ray.direction)

        if(abs(denom) > 0.0001):
            t = np.dot(self.origin - ray.origin, self.normal()) / denom
            if(t > 0):
                return(ray.origin + (ray.direction * t), t)

        return(None, None)


class Light:
    """A light is a sphere that is a bounce terminator"""
    def __init__(self, origin, radius, brightness, color):
        self.origin = origin
        self.radius = radius
        self.brightness = brightness
        self.color = np.array(color) / 255.0

    def intersect(self, ray, bounce):
        """No more bounces!"""
        return(geo_intersectsect(self, ray))

    def normal(self, origin):
        dir = origin - self.origin
        norm = np.linalg.norm(dir)
        if(norm != 0):
            dir /= norm
        return(dir)

# based on https://www.scratchapixel.com/lessons/mathematics-physics-for-computer-graphics/lookat-function/framing-lookat-function
# and https://www.youtube.com/watch?v=LRN_ewuN_k4
class Camera:

    def __init__(self, origin, target, aperture = 4.0, length = 1.0):
        self.origin = np.array(origin)
        self.target = np.array(target)
        self.aperture = aperture
        self.length = length

        forward = Point(self.target - self.origin)
        forward.normalize()

        right = Point(np.cross(forward.coords, np.array([0, 1, 0])))
        up = Point(np.cross(right.coords, forward.coords))

        self.rotation_matrix = np.array([
            right.coords, up.coords, forward.coords
        ])

class Scene:
    def __init__(self, filename = "scene.json"):

        self.lights = []
        self.objects = []

        # read in the json
        with open(filename, "r") as json_file:
            scene = json.load(json_file)

        for light in scene['scene']['lights']:
            self.lights.append(Light(origin = light['origin'], radius = light['radius'],
                                     brightness = light['brightness'], color = light['color']))

        for object in scene['scene']['objects']:
            if(object['shape'] == "Sphere"):
                self.objects.append(Sphere(origin = object['origin'], radius = object['radius'],
                                           diffuse = object['diffuse'], reflection = object['reflection'],
                                           shiny = object['shiny'], k = object['k'],
                                           color = object['color']))
            elif(object['shape'] == "Plane"):
                self.objects.append(Plane(origin = object['origin'], normal = object['normal'],
                                          diffuse = object['diffuse'], reflection = object['reflection'],
                                          shiny = object['shiny'], k = object['k'],
                                          color1 = object['color1'], color2 = object['color2']))

        camera = scene['scene']['camera']
        self.camera = Camera(origin = camera['origin'], target = camera['target'],
                             length = camera['length'], aperture = camera['aperture'])

        self.ambient = np.array(scene['scene']['ambient'])


    def render(self, size, rank, img_height, img_width, max_bounces):

      pixels = np.zeros((img_height, img_width, 3))
      zbuffer = np.full((img_height, img_width), None)
      rays = 0

      # displacement from the top left corner to the middle of the pixel
      PIXEL_WIDTH = (2.0 / (img_width + 1))
      PIXEL_HEIGHT = (2.0 / (img_height + 1))

      stats = {'rank': rank, 'pixels': 0, 'geo_intersections': 0, 'time': time.time()}

      for y in range(0, img_height):
          for x in range(0, img_width):

              rays += 1
              if(rays % (size - 1) == (rank - 1)):

                  stats['pixels'] = stats['pixels'] + 1

                  # create a ray from the default camera to this pixel
                  # camera location, (image plane x, image plane y, image plane z)
                  plane = np.array([((x + 1) * PIXEL_WIDTH) - 1, ((img_height - y + 1) * PIXEL_HEIGHT) - 1, self.camera.length])

                  ray = Ray(self.camera.origin, plane)
                  ray.normalize()

                  # now we apply the camera rotation to the ray direction
                  ray.direction = np.matmul(self.camera.rotation_matrix, ray.direction)
                  ray.normalize()

                  for o in self.objects:
                      pixel, depth, distance, geo_intersections = o.intersect(ray, self, 0, max_bounces)
                      stats['geo_intersections'] = stats['geo_intersections'] + geo_intersections
                      if(zbuffer[y][x] is None or (distance is not None and distance < zbuffer[y][x])):
                          pixels[y][x] = pixel.clip(0, 255)
                          zbuffer[y][x] = distance

      stats['time'] = time.time() - stats['time']
      return(pixels, stats)

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
         max_bounces = 3, show_stats = False, long_stats = False, sd = 2.0, input_filename = 'scene.json',
         camera = None, target = None):

  comm = MPI.COMM_WORLD
  rank = comm.Get_rank()
  size = comm.Get_size()

  scene = Scene(input_filename)

  if(camera is not None and target is not None):
      # override scene camera from command line args
      scene.camera = Camera(origin = np.array(camera, dtype=np.float32),
                       target = np.array(target, dtype=np.float32))

  if(rank != 0):
      pixels, stats = scene.render(size, rank, img_height, img_width, max_bounces)
      comm.Send(pixels, dest = 0)
      comm.send(stats, dest = 0)

  if(rank == 0):

      start = time.time()
      pixels = np.zeros((img_height, img_width, 3))
      stats = []

      for worker in range(1, size):
          worker_pixels = np.zeros((img_height, img_width, 3))
          comm.Recv(worker_pixels, source = worker)

          worker_stats = comm.recv(source = worker)

          pixels += worker_pixels
          stats.append(worker_stats)

      output = PIL.Image.fromarray(pixels.astype('uint8'), 'RGB')
      output.save(output_filename)

      print_ascii(output, 80)
      print("\033[0mThere... are... %i... lights!" %(len(scene.lights)))
      print("Finished in %f seconds." % (time.time() - start))

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
                  print(" - Worker %i, time: %f, pixels %i, intersections: %i." %(
                      worker['rank'],
                      worker['time'],
                      worker['pixels'],
                      worker['geo_intersections']
                  ))

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='A simple MPI raytracer.')

    parser.add_argument('--bounces', type = int , default = 50, help='max number of bounces')
    parser.add_argument('--height', type = int , default = 200, help='output image height')
    parser.add_argument('--width', type = int , default = 200, help='output image width')
    parser.add_argument('--cols', type = int , default = 200, help='terminal columns for ascii output')
    parser.add_argument('--sd', type = float , default = 2.0, help='number of standard deviations from the mean to consider a node an outlier')
    parser.add_argument('--output', type = str , default = 'output.png', help='output image filename')
    parser.add_argument('--scene', type = str , default = 'scene.json', help='json scene file')
    parser.add_argument('--stats', action='store_true', help='show stats on computational distribution')
    parser.add_argument('--long', action='store_true', help='show long stats on computational distribution')

    parser.add_argument('--camera', nargs="+", default=None, help = "camera origin")
    parser.add_argument('--target', nargs="+", default=None, help = "camera target")
    parser.add_argument('--focal', type = float , default = 1.0, help='camera focal length')
    parser.add_argument('--aperture', type = float , default = 2.0, help='camera aperture')

    args = parser.parse_args()
    main(args.height, args.width, args.cols, args.output, args.bounces,
         args.stats, args.long, args.sd, args.scene, args.camera, args.target)
