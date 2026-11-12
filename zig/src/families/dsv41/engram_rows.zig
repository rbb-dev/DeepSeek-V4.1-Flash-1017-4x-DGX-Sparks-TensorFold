    return workerSlotsN(units, if (pair) 2 else 1);
}

/// `workerSlots` for `lanes` passes in flight (TF_DS_ROCI_STAGGER_LANES: 3 takes three times the units, up to max_wk).
pub fn workerSlotsN(units: usize, lanes: usize) usize {
    return @max(2, if (lanes >= 2) lanes * units else units);

    chunk_dev_b: u64 = 0, // the second lane's pass: its own (0: one pass at a time)
    chunk_dev_c: u64 = 0, // the third lane's pass (TF_DS_ROCI_STAGGER_LANES=3; 0: two at most)
    wk: [max_wk][2]Stage, // a fill's worker's records, by chunk slot and Engram layer slot (prompt.engramWorker); the first wk_n slots exist
    wk_n: usize,
    chunk_dev_b: u64,
    chunk_dev_c: u64,
        for (0..2 * o.wk_slots) |i| {
            .wk_n = o.wk_slots,
            .chunk_dev_b = o.chunk_dev_b,
            .chunk_dev_c = o.chunk_dev_c,
        for (r.wk[0..r.wk_n]) |*row| {
    pub fn roundLand(r: *Rows, j: usize, rows: usize, s: cuda.Stream, out: u64) !void {
    /// A round's layer slot `j` that will not land (its round failed, G0.3): its submitted batch waited for and dropped
    /// (nothing kept), so the slot's staging and batch id are free for the next round. Errors are ignored.
    pub fn roundDrain(r: *Rows, j: usize) void {
        const st = &r.round[j];
        if (st.batch == 0) return;
        const id = st.batch;
        st.batch = 0;
        if (r.aio) |x| x.wait(id) catch {};
    }

    pub fn chunkApply(r: *Rows, st: *const Stage, n: usize, s: cuda.Stream, out: u64) !void {
        return r.chunkApplyOn(st, n, s, out, 0);
    }

    /// chunkApply on a lane: the second and third lanes' passes decode through their own device records, so passes in flight never share them.
    pub fn chunkApplyOn(r: *Rows, st: *const Stage, n: usize, s: cuda.Stream, out: u64, lane: u2) !void {
        const dev = switch (lane) {
            0 => r.chunk_dev,
            1 => r.chunk_dev_b,
            else => r.chunk_dev_c,
        };
        try upload(s, dev, st.bytes[0..bytes]);
        try r.kernels.decodeRows(s, dev, out, n, sl, if (r.owned) r.lo else 0, r.k, r.dim);