#!/bin/sh
# Dev helper (root): put the Ref2VA files that lie in /data/local/tmp (r2v: blk/adaln/lora/glue, vae: video VAE decoder, henc: image encoder) into the app's files/r2v by MOVING them (/data/local/tmp and the app data are different file systems: file by file, so the flash never holds a second full copy) and give them to the app.
# usage (on the phone, as root): sh install_r2v.sh [SRC_R2V=/data/local/tmp/r2v] [SRC_VAE=/data/local/tmp/vae] [SRC_ENC=/data/local/tmp/henc]
P=com.hexanima.npu; H=/data/data/$P; M=$H/files/r2v; R=${1:-/data/local/tmp/r2v}; V=${2:-/data/local/tmp/vae}; E=${3:-/data/local/tmp/henc}
[ -d $H/files ] || { echo "app data not found"; exit 1; }
mkdir -p $M/vae/glue $M/enc $M/lora
for f in $R/blk??.bin $R/adaln??.bin $R/r0.bin $R/r1.bin $R/hn.bin $R/H64.f32 $R/H256.f32 $R/vpp_w.f32 $R/vpp_b.f32 $R/cond_w.f32 $R/cond_b.f32 $R/te_in_w.f32 $R/te_in_b.f32 $R/te_out_w.f32 $R/te_out_b.f32 $R/fin_ada_w.f32 $R/fin_ada_b.f32 $R/fin_norm.f32 $R/vout_w.f32 $R/vout_b.f32 $R/rfin.f32 $R/rope_inv.f32; do [ -f $f ] && mv -f $f $M/ || echo "missing $f"; done
for f in $R/lora/*.bin; do mv -f $f $M/lora/; done
for f in $V/vb??.bin; do mv -f $f $M/vae/; done
for f in $V/glue/*; do mv -f $f $M/vae/glue/; done
mv -f $E/henc.bin $M/enc/henc.bin
UID_=$(stat -c %u $H); chown -R $UID_:$UID_ $M; chmod -R u+rwX,go-rwx $M; restorecon -R $M 2>/dev/null
touch $M/.installed; chown $UID_:$UID_ $M/.installed
ls $M | wc -l; du -sh $M
