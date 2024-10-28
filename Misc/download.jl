using Tar, GZip, Downloads, ProgressMeter

download_path = "https://zenodo.org/records/14002229/files/Data.tar.gz?download=1"

global p = nothing

function progress(total, now)
    if isnothing(p) && total > 0
        global p = Progress(total)
    end

    if !isnothing(p)
        update!(p, now)
    end
end

@info "Downloading data"
Downloads.download(download_path, "Data.tar.gz"; progress)

@info "Extracting data"
GZip.open("Data.tar.gz") do file
    Tar.extract(file, "Data/")
end