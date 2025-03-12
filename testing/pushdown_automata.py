import matplotlib.pyplot as plt
import matplotlib.patches as patches

class PDAStackVisualizer:
    def __init__(self, ax, initial_stack):
        self.ax = ax
        self.stack = list(initial_stack)
        self.stack_rects = []
        self.update_visualization()

    def push(self, symbol):
        self.stack.append(symbol)
        self.update_visualization()

    def pop(self):
        if self.stack:
            self.stack.pop()
            self.update_visualization()

    def update_visualization(self):
        self.clear_stack_rects()
        stack_height = len(self.stack)
        for i, symbol in enumerate(reversed(self.stack)):
            rect = patches.Rectangle((0.2, 0.1 + i * 0.2), 0.6, 0.18, facecolor='skyblue', edgecolor='black')
            self.ax.add_patch(rect)
            self.ax.text(0.5, 0.2 + i * 0.2, symbol, ha='center', va='center', fontsize=12)
            self.stack_rects.append(rect)
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, stack_height * 0.2 + 0.3)
        self.ax.axis('off')
        plt.pause(0.5)

    def clear_stack_rects(self):
      for rect in self.stack_rects:
        rect.remove()
      self.stack_rects = []


if __name__ == '__main__':
    fig, ax = plt.subplots()
    initial_stack = ['Z0']
    pda_stack_vis = PDAStackVisualizer(ax, initial_stack)

    pda_stack_vis.push('a')
    pda_stack_vis.push('a')
    pda_stack_vis.pop()
    pda_stack_vis.push('b')
    pda_stack_vis.pop()
    pda_stack_vis.pop()

    plt.show()
