using PythonCall

xiapi = pyimport("ximea.xiapi")

struct XimeaCamera
    camera::Py
    image::Py
end

function XimeaCamera()
    camera = xiapi.Camera()
    camera.open_device()
    camera.start_acquisition()

    XimeaCamera(camera, xiapi.Image())
end

function XimeaCamera(configs...)
    camera = XimeaCamera()

    for (key, value) ∈ configs
        set_param(camera, key, value)
    end

    camera
end

function capture!(buffer, camera::XimeaCamera)
    camera.camera.get_image(camera.image)
    py_img = camera.image.get_image_data_numpy()
    permutedims!(buffer, pyconvert(Matrix{UInt8}, py_img), (2, 1))
    reverse!(buffer, dims=2)
end

function capture(camera::XimeaCamera)
    width = pyconvert(Int, camera.camera.get_param("width"))
    height = pyconvert(Int, camera.camera.get_param("height"))
    buffer = Matrix{UInt8}(undef, width, height)
    capture!(buffer, camera)
end

function Base.close(camera::XimeaCamera)
    camera.camera.stop_acquisition()
    camera.camera.close_device()
    nothing
end

function set_param(camera::XimeaCamera, param::String, value)
    try
        camera.camera.set_param(param, value)
    catch
        @warn "Could not set the parameter $param to the value $value."
    end
    nothing
end

function get_param(camera::XimeaCamera, param::String)
    try
        camera.camera.get_param(param)
    catch
        @warn "Could not get the value of the parameter $param"
    end
end