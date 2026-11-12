    // The stand-in's nap ends at the deadline. A real WFET nap (aarch64) can end early at the event stream (~10 kHz, napOnce's
    // doc), so nap again until the deadline, as the proxy does: the one interval the instrument times is then the whole wait.
    var ran = false;
    while (nap.now() < dl) ran = nap.napOnce(&word, 0, dl);
    try std.testing.expect(ran);