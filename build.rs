//! Prost build script — compile Protobuf to Rust types (protobuf feature only)
use std::io::Result;

fn main() -> Result<()> {
    #[cfg(feature = "protobuf")]
    {
        let proto_dir = "schemas";
        prost_build::Config::new()
            .type_attribute(".", "#[derive(serde::Serialize, serde::Deserialize)]")
            .compile_protos(
                &[
                    &format!("{}/market_snapshot.proto", proto_dir),
                    &format!("{}/unified_order.proto", proto_dir),
                    &format!("{}/alpha_signal.proto", proto_dir),
                    &format!("{}/alphacast_output.proto", proto_dir),
                ],
                &[proto_dir],
            )?;
        println!("cargo:rerun-if-changed=schemas/market_snapshot.proto");
        println!("cargo:rerun-if-changed=schemas/unified_order.proto");
        println!("cargo:rerun-if-changed=schemas/alpha_signal.proto");
        println!("cargo:rerun-if-changed=schemas/alphacast_output.proto");
    }

    Ok(())
}
