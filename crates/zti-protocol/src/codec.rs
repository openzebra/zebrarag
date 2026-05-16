use anyhow::Result;
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};

pub async fn read_frame<R: AsyncRead + Unpin, T: serde::de::DeserializeOwned>(
    reader: &mut R,
) -> Result<T> {
    let mut len_buf = [0u8; 4];
    reader.read_exact(&mut len_buf).await?;
    let len = u32::from_le_bytes(len_buf) as usize;

    let mut payload = vec![0u8; len];
    reader.read_exact(&mut payload).await?;

    let msg: T = bincode::serde::decode_from_slice(&payload, bincode::config::standard())?.0;
    Ok(msg)
}

pub async fn write_frame<W: AsyncWrite + Unpin, T: serde::Serialize>(
    writer: &mut W,
    msg: &T,
) -> Result<()> {
    let payload = bincode::serde::encode_to_vec(msg, bincode::config::standard())?;
    let len = payload.len() as u32;
    writer.write_all(&len.to_le_bytes()).await?;
    writer.write_all(&payload).await?;
    writer.flush().await?;
    Ok(())
}
