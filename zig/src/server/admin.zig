    // what this key may do in this build: an admin key may write the rows of class H and A (live where the engine has a hook, A through a cfg frame) and R (engine.cfg)
    const classes = try a.alloc(Value, if (writes) 3 else 0);
        classes[1] = str("A");
        classes[2] = str("R");