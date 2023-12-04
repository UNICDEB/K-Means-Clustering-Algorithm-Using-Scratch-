import numpy as np
from sklearn.cluster import KMeans

def generate_anchors(annotations, num_clusters=9):
    """Generates anchor boxes using KMeans clustering in a custom dataset.
    
    Args:
        annotations: List of bounding box annotations in the form of [width, height].
        num_clusters: Number of anchor boxes to generate.
        
    Returns:
        anchor_boxes: List of anchor boxes in the form of [width, height].
    """
    
    # print("list Item :- ",annotations)
    # Convert annotations to a numpy array
    annotations = np.array(annotations)
    # print("After converting Array - \n", annotations)
    # print(type(annotations))
    
    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(annotations)
    anchor_boxes = kmeans.cluster_centers_
    
    print("All Cluster Value = \n",anchor_boxes)
    
    return anchor_boxes
# annotations=[[23,45],[34,78],[32,89],[13,25],[31,45],[32,78],[12,45],[29,90],[11,21],[31,45],[43,71],[32,78]]

# annotations=[[23,45],[34,78],[32,89],[13,25],[31,45],[32,78],[12,45],[29,90],[11,21],[31,45],[43,71],[32,78]]
csv_file = 'annotations_data.csv'
data_array = np.genfromtxt(csv_file, delimiter=',')
# print("Loaded data array:")
annotations = data_array
# print(data_array)
# print(type(annotations))
generate_anchors(annotations,num_clusters=9)