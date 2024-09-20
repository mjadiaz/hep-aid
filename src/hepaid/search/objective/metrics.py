import math 
import numpy as np
import pandas as pd
from typing import Any

from scipy.spatial.distance import cdist
from scipy.spatial.distance import pdist, squareform

from rich.table import Column
from rich.progress import Progress, BarColumn, TextColumn

import warnings

def in_jupyter_notebook():
    try:
        from IPython import get_ipython
        if 'IPKernelApp' not in get_ipython().config:
            return False
    except Exception:
        return False
    return True

class Metrics:
    """
    Class to track universal search metrics obtained from the Objective object.
    To add custom metrics related to algorithms, use the new_custom_metrics method and update 
    accordingly with a metric dictionary.

    Attributes:
        _metrics (dict): A dictionary to store universal metrics such as success rate, total points, valid points, satisfactory points, and iteration.
        _custom_metrics (dict): A dictionary to store custom metrics. Default is None.
        metrics_name (str): The name of the metrics file to save metrics data. Default is 'metrics.csv'.

    Methods:
        new_custom_metrics(names): Create keys and lists for new custom metrics.
        update_custom(custom_metrics_dict): Update custom metrics by appending new custom metrics values.
        update(objective_function, iteration): Update universal metrics by appending new metrics values.
        start_progress(description): Start a rich progress bar for logging progress.
        log(progress): Display metrics in the console. If used in a Jupyter notebook, display is turned off.
        save(save_path, iteration): Save metrics as a CSV dataset with the name 'metrics'.
        load(path): Load previous metrics from a CSV file.
    """
    def __init__(self):
        """
        Initializes the Metrics object with default universal metrics and prepares for custom metrics.
        """
        self._metrics = {
            "success_rate": [],
            "n_total_points": [],
            "n_valid_points": [],
            "n_satisfactory_points": [],
            "iteration": [],
        }

        self._custom_metrics = None
        self.metrics_name = 'metrics.csv'
    
    @property
    def metrics(self):
        """
        Property to access the universal metrics.

        Returns:
            dict: The dictionary containing universal metrics.
        """
        return self._metrics

    
    def new_custom_metrics(self, names: list[str]):
        """
        Create keys and lists for new custom metrics. If custom metrics are already loaded, do nothing.

        Parameters:
            names (list of str): List of names for the new custom metrics.
        """
        if self._custom_metrics is None:
            self._custom_metrics = {}
            for name in names:
                self._custom_metrics[name] = []

    def update_custom(self, custom_metrics_dict: dict[str, float | int]):
        """
        Update custom metrics by appending the new custom metrics values.

        Parameters:
            custom_metrics_dict (dict): A dictionary containing custom metric names as keys and their corresponding values.
        """
        for key in custom_metrics_dict:
            if key in self._custom_metrics:
                self._custom_metrics[key].append(custom_metrics_dict[key])

    def update(self, objective_function, iteration):
        """
        Update universal metrics by appending new metrics values.

        Parameters:
            objective_function (Objective): The Objective object to extract metrics from.
            iteration (int): The current iteration number.
        """
        if isinstance(objective_function.Y, type(None)):
            warnings.warn("Objective function without data")
            valid = successful = np.array([])
        else:
            valid = np.prod(~np.isnan(objective_function.Y), axis=1).astype(np.bool8)
            successful = objective_function.satisfactory.prod(axis=-1).astype(bool)

        self._metrics["success_rate"].append(successful.sum() / len(successful))
        self._metrics["n_valid_points"].append(valid.sum())
        self._metrics["n_total_points"].append(len(valid))
        self._metrics["n_satisfactory_points"].append(successful.sum())
        self._metrics["iteration"].append(iteration)
        
    def start_progress(self, description = "CAS search"):
        """
        Start a rich progress bar for logging progress.

        Parameters:
            description (str, optional): The description for the progress bar. Default is "CAS search".

        Returns:
            progress: A rich Progress object for logging progress.
        """
        text_column = TextColumn(description, table_column=Column(ratio=1))
        bar_column = BarColumn(bar_width=None, table_column=Column(ratio=2))
        progress = Progress(text_column, bar_column, expand=True)
        return progress

    def log(self, progress):
        """
        Display metrics in the console. If used in a Jupyter notebook, display is turned off.

        Parameters:
            progress (Progress): The rich Progress object used for logging.
        """
        if in_jupyter_notebook():
            return
        else:
            if self._custom_metrics is not None:
                total_metrics = {**self._metrics, **self._custom_metrics}
            else:
                total_metrics = self._metrics
            output = ""
            for key, value in total_metrics.items():
                if value:
                    last_element = value[-1]
                    output += f"{key}: {last_element}\n"
            progress.print(output)
    
    def save(self, save_path, iteration):
        """
        Save metrics as a CSV dataset with the name 'metrics'.

        Parameters:
            save_path (Path): The path to save the metrics CSV file.
            iteration (int): The current iteration number.
        """
        if self._custom_metrics is not None:
            total_metrics = {**self._metrics, **self._custom_metrics}
        else:
            total_metrics = self._metrics
        metrics_df = pd.DataFrame(total_metrics)
        metrics_df.to_csv(save_path / self.metrics_name, index=False)
    
    def load(self, path):
        """
        Load previous metrics from a CSV file.

        Parameters:
            path (Path): The path to the metrics CSV file.
        """
        metrics_df = pd.read_csv(path / self.metrics_name , index_col=None)
        metrics_dict = metrics_df.to_dict(orient='list')
        for m in self._metrics.keys():
            self._metrics[m] = metrics_dict.pop(m)
        if len(metrics_dict.keys()) != 0:
            self._custom_metrics = metrics_dict


class MeanEuclideanDistance:
    def __init__(self, x_dim=2):
        self.dimension = x_dim
        self.points = []
        self._history = []
        self.counter = 0
    
    @property
    def history(self):
        return np.array(self._history)
    
    @property
    def distances(self):
        return self._distances
    
    @property
    def distance_matrix(self):
        return squareform(self._distances)
    
    def add(self, point):
        if self.counter == 0:
            self.points = np.array(point).reshape(-1, self.dimension)
        else:
            self.points = np.vstack((self.points,point.reshape(1, -1)))  

            distances = pdist(self.points)
            mean_distance = distances.sum()/len(distances)
            self._history.append(mean_distance)
            self._distances = distances
        
        self.counter += 1

def generate_random_points_in_sphere(n, r, dimensions):
    # Generate random unit vectors
    random_unit_vectors = np.random.normal(size=(n, dimensions))
    random_unit_vectors /= np.linalg.norm(random_unit_vectors, axis=1)[:, np.newaxis]
    
    # Generate random radii
    random_radii = np.power(np.random.uniform(0, 1, n), 1/dimensions) * r
    
    # Convert to Cartesian coordinates
    random_points = random_radii[:, np.newaxis] * random_unit_vectors
    
    return random_points

def n_dimensional_sphere_volume(n, radius):
    if n % 2 == 0:
        # For even dimensions, use the formula directly
        volume = (math.pi**(n/2) / math.gamma(n/2 + 1)) * radius**n
    else:
        # For odd dimensions, use the formula for the surface area of an n-1 dimensional sphere
        # and multiply by the height (radius) to get the volume
        surface_area = (2 * math.pi**(n/2) / math.gamma(n/2)) * radius**(n-1)
        volume = surface_area * radius

    return volume

 
class VolumeCoverage:
    def __init__(
        self, 
        radius: float | None = None,
        x_dim: int = 2, 
        points_per_sphere: int = 500
        ):
        self.dimension = x_dim
        self.radius = radius
        self.points_per_sphere = points_per_sphere

        self.points = []
        self._spheres = []
        self._history = []
        self._cumulative = []

        self.counter = 0
    
    @property
    def history(self) -> np.ndarray:
        '''History of volumes per added sphere.
        '''
        return np.array(self._history)
    
    @property 
    def spheres(self) -> list:
        '''History of samples per added sphere.
        '''
        return self._spheres

    @property
    def cumulative(self) -> np.ndarray:
        '''Utility for cumulative volume covered.
        '''
        return np.array(self._cumulative)

    def add(self, point: np.ndarray, radius: float | None = None) -> None:
        '''
        If is the first just add the first set of points per sphere. Then 
        calculate if any of the previous spheres might overlap with the new 
        point sphere. For each overlapping sphere, remove the points of the new 
        sphere that overlap. For the remaining points P_r, calculate the ratio 
        of the original total `points_per_sphere` used to sample the sphere. Multiply 
        the ratio with the exact volume. 
        '''
        if radius == None:
            radius = self.radius
        if self.counter == 0:
            
            ball = generate_random_points_in_sphere(
                self.points_per_sphere, radius, self.dimension
                )
            ball = ball + point
            self.points = np.array(point).reshape(-1, self.dimension)
            self._spheres.append(ball)
            self._history.append(
                n_dimensional_sphere_volume(2, radius)
                )
        else:
            overlapping_distances = cdist(self.points, point.reshape(-1,self.dimension))
            is_overlapped = (overlapping_distances < 2*radius).flatten()
            ball = generate_random_points_in_sphere(
                self.points_per_sphere, radius, self.dimension
                )
            ball = ball + point
            for i in np.where(is_overlapped)[0]:
                if len(ball) == 0:
                    break
                distances = cdist(ball,self.points[i].reshape(1, -1))
                select = distances.flatten() < radius
                ball = ball[~select]
            
            # Volume by formula and ratio
            ratio = len(ball)/self.points_per_sphere
            exact_volume = n_dimensional_sphere_volume(2, radius)
            self._history.append(exact_volume*ratio + 1e-8)
            
            self._spheres.append(ball) 
        
            self.points = np.vstack((self.points,point.reshape(1, -1))) 
           

            
        self._cumulative.append(
            sum(
                self._history
            )
        )
        self.counter += 1
            
            
def s_ns_history(
    volumes: np.ndarray, tau: np.ndarray[bool] | list[bool]
    )-> tuple[np.ndarray, np.ndarray]:
    '''
    Cummulative satisfactory and non-Satisfactory history given a bolean array.
    '''
    sat_vol = []
    non_sat_vol = []

    sat_vol_value = 0
    non_sat_vol_value = 0
    for v, t in zip(volumes, tau):
        if t:
            sat_vol_value += v
        else:
            non_sat_vol_value += v

        sat_vol.append(sat_vol_value)
        non_sat_vol.append(non_sat_vol_value)
    return np.array(sat_vol), np.array(non_sat_vol)
            
if __name__ == "__main__":
    pass
   # vc = VolumeCoverage(0.01)
   # points = np.random.uniform(0,1,(1000,2))
   # for p in points:
   #     vc.add(p)
   #     print(vc.cumulative[-1])
    