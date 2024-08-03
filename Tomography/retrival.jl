using StructuredLight, HDF5, CairoMakie, BayesianTomography

path = "Data/Raw/Old/blade.h5"

fit_param, x, y = h5open(path) do file
    obj = file["fit_param"]
    obj |> read, attrs(obj)["x"], attrs(obj)["y"]
end

