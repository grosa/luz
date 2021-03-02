# a delta takes a named object and applies a dictionary of changes to it
class Delta:
    def __init__(self, obj, changes):
        self.obj = obj
        self.changes = changes

    # changes object in scene file
    def apply(self, scene):
        pass

class Frame:
    def __init__(self, number, deltas):
        self.number = number
        self.deltas = deltas

    def apply(self, scene):
        for delta in self.deltas
            scene = delta.apply(scene)
        self.scene = scene
        return scene

    def render(self):
        self.scene.render()

class Animation:
    def __init__(self, scene, frames)
        self.scene = scene

    def render(self):
        for frame in frames:
            scene = frame.apply(scene)
            frame.render()
