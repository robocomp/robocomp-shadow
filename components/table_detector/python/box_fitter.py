try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from pytorch3d.loss import chamfer_distance
    from pytorch3d.structures import Pointclouds
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
except ModuleNotFoundError as e:
    print("Required module not found. Ensure PyTorch and PyTorch3D are installed.")
    raise e


def plot_pointclouds(real_pc, synthetic_pc):
    """
    Plot the real and synthetic point clouds in the same graph.
    """
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot real point cloud
    ax.scatter(real_pc[:, 0].cpu().detach().numpy(),
               real_pc[:, 1].cpu().detach().numpy(),
               real_pc[:, 2].cpu().detach().numpy(),
               c='b', label='Real Point Cloud', s=1)

    # Plot synthetic point cloud
    ax.scatter(synthetic_pc[:, 0].cpu().detach().numpy(),
               synthetic_pc[:, 1].cpu().detach().numpy(),
               synthetic_pc[:, 2].cpu().detach().numpy(),
               c='r', label='Synthetic Point Cloud', s=1)

    # Set labels and title
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Real vs Synthetic Point Clouds")
    ax.legend()

    plt.show()

# ===========================
# Step 1: Load the Real Point Cloud
# ===========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

try:
    real_pc = torch.load("real_fridge_pointcloud.pt").to(device)
except FileNotFoundError:
    print("Error: real_fridge_pointcloud.pt not found. Ensure the file exists.")
    raise


# ===========================
# Step 2: Define the Fridge Model
# ===========================
class FridgeModel(nn.Module):
    def __init__(self, init_params, device):
        super().__init__()
        self.device = device

        # Define each parameter separately as a trainable nn.Parameter
        self.x = nn.Parameter(torch.tensor(init_params[0], dtype=torch.float32, requires_grad=True, device=self.device))
        self.y = nn.Parameter(torch.tensor(init_params[1], dtype=torch.float32, requires_grad=True, device=self.device))
        self.z = nn.Parameter(torch.tensor(init_params[2], dtype=torch.float32, requires_grad=True, device=self.device))

        self.a = nn.Parameter(torch.tensor(init_params[3], dtype=torch.float32, requires_grad=True, device=self.device))
        self.b = nn.Parameter(torch.tensor(init_params[4], dtype=torch.float32, requires_grad=True, device=self.device))
        self.c = nn.Parameter(torch.tensor(init_params[5], dtype=torch.float32, requires_grad=True, device=self.device))

        self.w = nn.Parameter(torch.tensor(init_params[6], dtype=torch.float32, requires_grad=True, device=self.device))
        self.d = nn.Parameter(torch.tensor(init_params[7], dtype=torch.float32, requires_grad=True, device=self.device))
        self.h = nn.Parameter(torch.tensor(init_params[8], dtype=torch.float32, requires_grad=True, device=self.device))

    def forward(self):
        corners = self.compute_orthohedron(self.x, self.y, self.z, self.a, self.b, self.c, self.w, self.d, self.h)
        synthetic_pc = self.sample_surface_points(corners)
        return synthetic_pc

    def compute_orthohedron(self, x, y, z, a, b, c, w, d, h):
        """Compute 3D corners of the oriented bounding box with proper gradient tracking."""
        base_corners = torch.stack([
            torch.tensor([-0.5, -0.5, -0.5], device=self.device) * torch.stack([w, d, h]),
            torch.tensor([0.5, -0.5, -0.5], device=self.device) * torch.stack([w, d, h]),
            torch.tensor([0.5, 0.5, -0.5], device=self.device) * torch.stack([w, d, h]),
            torch.tensor([-0.5, 0.5, -0.5], device=self.device) * torch.stack([w, d, h]),
            torch.tensor([-0.5, -0.5, 0.5], device=self.device) * torch.stack([w, d, h]),
            torch.tensor([0.5, -0.5, 0.5], device=self.device) * torch.stack([w, d, h]),
            torch.tensor([0.5, 0.5, 0.5], device=self.device) * torch.stack([w, d, h]),
            torch.tensor([-0.5, 0.5, 0.5], device=self.device) * torch.stack([w, d, h])
        ])

        R = self.euler_to_rotation_matrix(a, b, c)
        rotated_corners = (R @ base_corners.T).T

        translation = torch.stack([x, y, z])
        return rotated_corners + translation

    def euler_to_rotation_matrix(self, a, b, c):
        """Convert Euler angles to a rotation matrix while maintaining gradient flow."""
        cos_a, sin_a = torch.cos(a), torch.sin(a)
        cos_b, sin_b = torch.cos(b), torch.sin(b)
        cos_c, sin_c = torch.cos(c), torch.sin(c)

        Rx = torch.stack([
            torch.stack([torch.tensor(1.0, device=a.device), torch.tensor(0.0, device=a.device), torch.tensor(0.0, device=a.device)]),
            torch.stack([torch.tensor(0.0, device=a.device), cos_a, -sin_a]),
            torch.stack([torch.tensor(0.0, device=a.device), sin_a, cos_a])
        ])

        Ry = torch.stack([
            torch.stack([cos_b, torch.tensor(0.0, device=b.device), sin_b]),
            torch.stack([torch.tensor(0.0, device=b.device), torch.tensor(1.0, device=b.device), torch.tensor(0.0, device=b.device)]),
            torch.stack([-sin_b, torch.tensor(0.0, device=b.device), cos_b])
        ])

        Rz = torch.stack([
            torch.stack([cos_c, -sin_c, torch.tensor(0.0, device=c.device)]),
            torch.stack([sin_c, cos_c, torch.tensor(0.0, device=c.device)]),
            torch.stack([torch.tensor(0.0, device=c.device), torch.tensor(0.0, device=c.device), torch.tensor(1.0, device=c.device)])
        ])

        return Rz @ Ry @ Rx  # Correct rotation order (Z-Y-X intrinsic)

    def sample_surface_points(self, corners, num_samples=1000):
        """Generate a synthetic point cloud by sampling points from the fridge model."""
        edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
                 (0, 4), (1, 5), (2, 6), (3, 7)]

        sampled_points = []
        for edge in edges:
            p1, p2 = corners[edge[0]], corners[edge[1]]
            t = torch.linspace(0, 1, num_samples // len(edges), device=self.device)
            points = p1 * (1 - t[:, None]) + p2 * t[:, None]
            sampled_points.append(points)

        return torch.cat(sampled_points, dim=0)

def print_grad(grad):
    print("Gradient:", grad)

# ===========================
# Step 3: Optimization Setup
# ===========================
init_params = [0.0, 0.3, 0.2, 0.0, 0.0, 0.0, 0.3, 0.3, 1.2]  # Initial guess
fridge_model = FridgeModel(init_params, device).to(device)
#fridge_model.params.register_hook(print_grad)
#optimizer = optim.Adam(fridge_model.parameters(), lr=0.01)
optimizer = optim.Adam([
    {'params': [fridge_model.x, fridge_model.y, fridge_model.z], 'lr': 0.01},  # Position
    {'params': [fridge_model.a, fridge_model.b, fridge_model.c], 'lr': 0.1},  # Rotation (higher LR)
    {'params': [fridge_model.w, fridge_model.d, fridge_model.h], 'lr': 0.01}   # Size
])
# ===========================
# Step 4: Optimize the Model
# ===========================
num_iterations = 1000
for i in range(num_iterations):
    optimizer.zero_grad()

    synthetic_pc = fridge_model()

    # Compute Chamfer Distance
    loss, _ = chamfer_distance(Pointclouds([synthetic_pc]), Pointclouds([real_pc]))

    loss.backward()

    optimizer.step()

    if i % 100 == 0:
        print(f"Iteration {i}: Loss = {loss.item()}")  # Check gradients
        #print("Gradients:", fridge_model.params.grad)
        #print(fridge_model.params.detach().cpu().numpy())

print("Optimization complete.")
plot_pointclouds(real_pc, synthetic_pc)
# print model params
print("Model Parameters:")
print("x: {:.4f}".format(fridge_model.x.item()))
print("y: {:.4f}".format(fridge_model.y.item()))
print("z: {:.4f}".format(fridge_model.z.item()))
print("a: {:.4f}".format(fridge_model.a.item()))
print("b: {:.4f}".format(fridge_model.b.item()))
print("c: {:.4f}".format(fridge_model.c.item()))
print("w: {:.4f}".format(fridge_model.w.item()))
print("d: {:.4f}".format(fridge_model.d.item()))
print("h: {:.4f}".format(fridge_model.h.item()))



