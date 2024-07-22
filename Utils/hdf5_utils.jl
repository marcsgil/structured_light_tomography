using HDF5

function safe_write(object_name, file_path, data)
    h5open(file_path, "cw") do file
        if object_name ∈ keys(file)
            @warn "Could not write: $object_name is already present in the file."
        else
            file[object_name] = object_name
        end
    end
end