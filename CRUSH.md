## fit-rs

This project uses Rust, so we'll use `cargo` for most tasks.

### Dependencies

Install dependencies with:
`cargo build`

### Building

Build the project with:
`cargo build --release`

This will create two binaries in `target/release`: `fit-rs` and `fit-server`.

### Running

Run the main binary with:
`cargo run --bin fit-rs`

Run the server with:
`cargo run --bin fit-server`

### Testing

Run all tests with:
`cargo test`

Run a single test with:
`cargo test --test <TEST_NAME>`

### Linting

Lint the project with:
`cargo clippy`

### Formatting

Format the code with:
`cargo fmt`
