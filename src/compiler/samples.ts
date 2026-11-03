// The two translation units compiled separately. The linker has real work to do
// because main.rs only *declares* square; math.rs defines it.
export const MATH_RS = `// math.rs: its own translation unit
fn square(x: i32) -> i32 {
    x * x
}
`;

export const MAIN_RS = `extern fn square(x: i32) -> i32;

// sum of squares 1..4, plus a folded constant
fn main() -> i32 {
    let mut sum: i32 = 0;
    let mut i: i32 = 1;
    let unused: i32 = 7 * 6;
    while i < 5 {
        sum = sum + square(i);
        i = i + 1;
    }
    let k: i32 = 2 + 3;
    sum + k
}
`;

export const EXPECTED_EXIT_CODE = 35; // 1 + 4 + 9 + 16 + 5
