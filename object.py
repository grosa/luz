import numpy as np
from ray import Ray

class Object:

    def collides_with(self, ray):
        origin, distance = self.geo_intersect(ray)

        if(origin is None or distance is None):
            return(False, None)
        return(True, distance)

    def intersect(self, ray, scene, bounce, max_bounces):

        origin, distance = self.geo_intersect(ray)
        total_geo_intersections = 1

        if(origin is None):
            return(scene.background, None, None, 1)

        normal = self.normal_at(origin.coords)
        color = np.full(3, 0.0)
        reflected_color = np.full(3, 0.0)

        for l in scene.lights:

            lightray = Ray(l.origin, l.origin - origin.coords)
            light_distance = lightray.distance()
            lightray.normalize()

            shadowray = Ray(origin.coords, l.origin - origin.coords)
            shadowray.normalize()

            in_shadow, shadow_intersections = shadowray.in_shadow(self, l, scene)
            total_geo_intersections = total_geo_intersections + shadow_intersections

            if in_shadow == False:
                color += (self.color_at(origin) * l.color * l.brightness * self.diffuse * max(np.dot(normal, lightray.direction), 0) / light_distance).clip(0, 255)

                if(self.shiny > 0.0):
                    cam_ray = Ray(origin.coords, scene.camera.origin - origin.coords)
                    cam_ray.normalize()

                    halfway = cam_ray.direction + lightray.direction
                    color += (l.color * l.brightness * self.shiny * max(np.dot(normal, halfway), 0) ** self.k / light_distance).clip(0, 255)

        if(self.reflection > 0.0 and bounce < max_bounces):
            dot = -np.dot(normal, ray.direction)
            raydir = ray.direction + (2 * dot * normal)
            newray = Ray(origin.coords, raydir)

            closest = 999
            for obj in scene.objects:
                if obj != self:
                    r_value, r_depth, r_distance, geo_intersections = obj.intersect(newray, scene, bounce + 1, max_bounces)
                    total_geo_intersections += geo_intersections
                    if(r_distance is not None and r_distance < closest):
                        closest = r_distance
                        reflected_color = r_value * self.reflection

        return((color + reflected_color + scene.ambient).clip(0, 255), origin.z(), distance, total_geo_intersections)
