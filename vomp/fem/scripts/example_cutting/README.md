# Topo cutting example


## Requirements

The following additional Python packages must be installed

 - polyscope==2.1.*
 - [kaolin](https://github.com/NVIDIAGameWorks/kaolin)

This example depends on the NVIDIA kaolin FlexiCubes implementation, which is part of the namespace `kaolin.non_commercial` released under the [NSCL license](https://github.com/NVIDIAGameWorks/kaolin/blob/master/LICENSE.NSCL).

## Usage

```
python example_cutting.py /path/to/mesh.obj [-qm quadrature_model.pt]
```

Run without options to see full list of possible arguments

Interactive commands:

 - Ctrl+left mouse: add material
 - Ctrl+right mouse: remove material
 - Shift+left drag: apply picking force

 
 