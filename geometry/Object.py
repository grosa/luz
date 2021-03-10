import numpy as np
from vectors.Ray import Ray

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
            return(scene.background, None, 1)

        normal = self.normal_at(origin.coords)
        color = np.full(3, 0.0)
        reflected_color = np.full(3, 0.0)
        refracted_color = np.full(3, 0.0)

        for l in scene.lights:

            lightray = Ray(l.origin, l.origin - origin.coords)
            light_distance = lightray.distance()
            lightray.normalize()

            shadowray = Ray(origin.coords, l.origin - origin.coords)
            shadowray.normalize()

            in_shadow = False
            shadow_intersections = 0

            if type(self).__name__ != "Skybox":
                in_shadow, shadow_intersections = shadowray.in_shadow(self, l, scene)

            total_geo_intersections = total_geo_intersections + shadow_intersections

            if in_shadow == False:
                # color += (self.color_at(origin) * l.color * l.brightness * self.diffuse * max(np.dot(normal, lightray.direction), 0) / light_distance).clip(0, 255)
                color += (self.color_at(origin) * l.color * l.brightness * self.diffuse * max(np.dot(normal, lightray.direction), 0) / light_distance)

                if(self.shiny > 0.0):
                    cam_ray = Ray(origin.coords, scene.camera.origin - origin.coords)
                    cam_ray.normalize()

                    halfway = cam_ray.direction + lightray.direction
                    # color += (l.color * l.brightness * self.shiny * max(np.dot(normal, halfway), 0) ** self.k / light_distance).clip(0, 255)
                    color += (l.color * l.brightness * self.shiny * max(np.dot(normal, halfway), 0) ** self.k / light_distance)


        if(self.refraction > 0.0 and bounce < max_bounces):

            n = self.normal_at(origin.coords)
            cosi = np.dot(ray.direction, n).clip(-1, 1)
            etai = 1.0 # air index
            etat = self.index # index of refraction

            if(cosi < 0):
                cosi = -cosi
            else:
                n = -n
                temp = etai
                etai = etat
                etat = temp

            eta = etai / etat
            k = 1 - eta * eta * (1 - cosi * cosi)

            if(k > 0):

                refraction_ray = Ray(origin.coords, eta * ray.direction + (eta * cosi - np.sqrt(k)) * n)
                refraction_ray.offset()

                closest = 999
                for obj in scene.objects:
                    # if obj != self:
                    r_value, r_distance, geo_intersections = obj.intersect(refraction_ray, scene, bounce + 1, max_bounces)
                    total_geo_intersections += geo_intersections
                    if(r_distance is not None and r_distance < closest):
                        closest = r_distance
                        refracted_color = (r_value * self.refraction) + self.color_at(origin) * self.diffuse

        if(self.reflection > 0.0 and bounce < max_bounces):
            dot = -np.dot(normal, ray.direction)
            raydir = ray.direction + (2 * dot * normal)
            newray = Ray(origin.coords, raydir)

            closest = 999
            for obj in scene.objects:
                if obj != self:
                    r_value, r_distance, geo_intersections = obj.intersect(newray, scene, bounce + 1, max_bounces)
                    total_geo_intersections += geo_intersections
                    if(r_distance is not None and r_distance < closest):
                        closest = r_distance
                        reflected_color = r_value * self.reflection

        # return((color + reflected_color + scene.ambient).clip(0, 255), origin.z(), distance, total_geo_intersections)
        return((color + reflected_color + + refracted_color + scene.ambient), distance, total_geo_intersections)
