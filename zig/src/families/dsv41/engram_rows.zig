    /// A round's layer slot `j` that will not land (its round failed, G0.3): its submitted batch waited for and dropped
    /// (nothing kept), so the slot's staging and batch id are free for the next round. Errors are ignored.
    pub fn roundDrain(r: *Rows, j: usize) void {
        const st = &r.round[j];
        if (st.batch == 0) return;
        const id = st.batch;
        st.batch = 0;
        if (r.aio) |x| x.wait(id) catch {};
    }
