# STEP/STL/OBJ converter script to OFF

### Dependencies
Install the dependencies:

* trimesh
* open3d
* cadquery (to convert STEP files)

## Quick Start
To run the model for a given object follow the below command.

```
python convertitore.py {object_name}.{object-extension}
```
For example:
```
python convertitore.py finito.step
```

\
You can also use this script to batch convert several objects.

```
python convertitore.py --batch {objects_directory} -o {OFF_output_directory}
```
For example:
```
python convertitore.py --batch ./models -o ./output_off
```
