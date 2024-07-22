function loop_capture!(output, desireds, incoming, slm, camera,
    x, y, max_modulation, xperiod, yperiod; sleep_time=0.15)
    N = size(desireds, 3)
    holo = generate_hologram(view(desireds, :, :, 1), incoming, x, y, max_modulation, xperiod, yperiod)
    @showprogress for n ∈ 1:N-1
        update_hologram(slm, holo; sleep_time)
        holo = generate_hologram(view(desireds, :, :, n + 1), incoming, x, y, 82, 5, 4)
        capture!(view(output, :, :, n), camera)
    end

    update_hologram(slm, holo)
    capture!(view(output, :, :, N), camera)
    nothing
end