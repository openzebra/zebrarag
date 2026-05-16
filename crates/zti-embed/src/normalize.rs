pub fn normalize_l2(v: &mut [f32]) {
    let norm_sq: f32 = v.iter().map(|x| x * x).sum();
    if norm_sq > 0.0 {
        let inv = norm_sq.sqrt().recip();
        for x in v.iter_mut() {
            *x *= inv;
        }
    }
}
