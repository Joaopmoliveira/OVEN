## OVEN 

This project is the implementation of the OVEN framework: a methodology used to extract the topology (area, orientation, inclination) of buildings from satellite images. 

We achieve this by creating a dataset with 3D information and forcing the Deep Neural Network (based on YOLO v11) to learn features that indicate the topology just by observing satellite images.

The basic idea behind this framework is to show a large collection of building topologies, as illustrated here:
![OVENFigure](images/ovengraphicalabstract.png)
Through back-propagation, we teach the network to find reliable estimates of the orientation and inclination of rooftops that can be used for solar panel installation.

To deal with variable building topologies (e.g., varying number of tiles), the deep model is forced to learn how to map buildings of arbitrary shape—as seen in the center of the following figure—into a convex representation, as seen on the right:
![OVENFigure](images/building_topology.png)

# Dataset

The curated dataset can be found in /dataset with images and labels split between training and validation data. 
Citation

# Citation

This code accompanies the journal submission of OVEN. If you find this project helpful and useful, please cite our work! It's appreciated!
