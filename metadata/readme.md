
## df_meta

- contains metadata info from the header of each UID as well as their file locations

## df_lesions

- Returned all center of mass from each uid

## df_lesions_erosion

- Applied erosion and returned all center of mass from each uid


## df_coords

- Applied erosion once and twice
- Got center of mass and bounding box around lesion
- Sorted by volume of bbox by CT and clipped by a threshold
- Randomly generated more coordinates within generated volumes using a gaussian distribution

## df_coords_debias

- Same steps as df_coords
- Applied upsampling so each UID will have the same number of coordinates
