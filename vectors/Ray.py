import numpy as np

class Ray:
    def __init__(self, origin, direction):
        self.origin = np.array(origin)
        self.direction = np.array(direction)

    def offset(self):
        self.origin = self.origin + self.direction * 0.01

    def distance(self):
        a = self.direction[0] * self.direction[0]
        b = self.direction[1] * self.direction[1]
        c = self.direction[2] * self.direction[2]

        return(np.sqrt(a+b+c))

    def normalize(self):
      norm = np.linalg.norm(self.direction)
      if norm != 0:
          self.direction /= norm

    def in_shadow(self, this, l, scene):
        intersections = 0
        # calculate distance from point to light
        light_distance = np.sqrt( sum( (l.origin - self.origin) * (l.origin - self.origin)) )

        for obj in scene.objects:
            if obj is not this:
                intersections = intersections + 1
                collided, distance = obj.collides_with(self)
                if(collided and distance < light_distance):
                    return(True, intersections)
        return(False, intersections)
