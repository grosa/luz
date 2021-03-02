import numpy as np

class Point:

    def __init__(self, coords, uvw):
        self.coords = coords
        self.uvw = uvw

    def x(self):
        return self.coords[0]

    def y(self):
        return self.coords[1]

    def z(self):
        return self.coords[2]

    def barycentric(self):
        return(self.uvw[0], self.uvw[1], self.uvw[2])

    def normalize(self):
      norm = np.linalg.norm(self.coords)
      if norm != 0:
          self.coords = self.coords / norm
