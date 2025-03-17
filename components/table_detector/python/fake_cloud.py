import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Set device (use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def plot_pointclouds(real_pc):
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

    # Set labels and title
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Real Point Clouds")
    ax.legend()

    plt.show()

def euler_to_rotation_matrix(a, b, c):
    """Convert Euler angles to a rotation matrix."""
    a, b, c = torch.tensor([a, b, c], dtype=torch.float32, device=device)
    Rx = torch.tensor([[1, 0, 0], [0, torch.cos(a), -torch.sin(a)], [0, torch.sin(a), torch.cos(a)]], dtype=torch.float32, device=device)
    Ry = torch.tensor([[torch.cos(b), 0, torch.sin(b)], [0, 1, 0], [-torch.sin(b), 0, torch.cos(b)]], dtype=torch.float32, device=device)
    Rz = torch.tensor([[torch.cos(c), -torch.sin(c), 0], [torch.sin(c), torch.cos(c), 0], [0, 0, 1]], dtype=torch.float32, device=device)
    return Rz @ Ry @ Rx

def compute_orthohedron(x, y, z, a, b, c, w, d, h):
    """
    Compute 3D corners of the oriented bounding box given parameters.
    """
    # Define the corners of the box in local coordinates
    base_corners = torch.tensor([
        [-w / 2, -d / 2, -h / 2],
        [w / 2, -d / 2, -h / 2],
        [w / 2, d / 2, -h / 2],
        [-w / 2, d / 2, -h / 2],
        [-w / 2, -d / 2, h / 2],
        [w / 2, -d / 2, h / 2],
        [w / 2, d / 2, h / 2],
        [-w / 2, d / 2, h / 2]
    ], dtype=torch.float32, device=device)

    # Compute rotation matrix
    R = euler_to_rotation_matrix(a, b, c)
    rotated_corners = (R @ base_corners.T).T

    # Convert x, y, z to tensors and stack them
    translation = torch.tensor([x, y, z], dtype=torch.float32, device=device)

    # Translate the corners
    return rotated_corners + translation

def sample_points_on_edges(corners, num_samples=1000):
    """Generate a synthetic point cloud by sampling points from the fridge model."""
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
             (0, 4), (1, 5), (2, 6), (3, 7)]

    sampled_points = []
    for edge in edges:
        p1, p2 = corners[edge[0]], corners[edge[1]]
        t = torch.linspace(0, 1, num_samples // len(edges), device=device)
        points = p1 * (1 - t[:, None]) + p2 * t[:, None]
        sampled_points.append(points)

    return torch.cat(sampled_points, dim=0)


# Save the fake point cloud to a file
def save_fake_pointcloud(filename, points):
    """
    Save the fake point cloud to a file.
    """
    torch.save(points, filename)


# Main script
if __name__ == "__main__":
    # Generate a fake fridge point cloud
    corners = compute_orthohedron(0.5, 0, 0, 0.0, 0.0, 0.3, 0.6, 0.6, 1.8)
    fake_fridge_points = sample_points_on_edges(corners)

    # Save the fake point cloud to a file
    filename = "real_fridge_pointcloud.pt"
    save_fake_pointcloud(filename, fake_fridge_points)

    print(f"Fake fridge point cloud saved to {filename}")
    plot_pointclouds(fake_fridge_points)
