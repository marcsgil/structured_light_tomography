using PrettyTables, HDF5

get_sigdigits(x) = -floor(Int, log10(x))

fid, blade_pos = h5open("Results/blade.h5") do file
    read(file["fid"]), read(file["blade_pos"])
end

fid

sigdigits = get_sigdigits.(fid[3, :] - fid[2, :])

data = [round(point; sigdigits) for (point, sigdigits) ∈ zip(fid[1,:], sigdigits)]

string()