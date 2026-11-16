package com.hexanima.npu;

import android.content.ContentProvider;
import android.content.ContentValues;
import android.database.Cursor;
import android.net.Uri;
import android.os.ParcelFileDescriptor;
import java.io.File;
import java.io.FileNotFoundException;

/** Read-only provider so that history pictures (private files) can be shared / opened by other apps. */
public class GalleryProvider extends ContentProvider {
    @Override public boolean onCreate() { return true; }
    @Override public ParcelFileDescriptor openFile(Uri u, String mode) throws FileNotFoundException {
        String n = u.getLastPathSegment();
        if (n == null || n.contains("/") || !(n.endsWith(".png") || n.endsWith(".mp4"))) throw new FileNotFoundException(String.valueOf(u));
        return ParcelFileDescriptor.open(new File(Gallery.dir(getContext()), n), ParcelFileDescriptor.MODE_READ_ONLY);
    }
    @Override public String getType(Uri u) { String n = u.getLastPathSegment(); return n != null && n.endsWith(".mp4") ? "video/mp4" : "image/png"; }
    @Override public Cursor query(Uri u, String[] p, String s, String[] a, String o) { return null; }
    @Override public Uri insert(Uri u, ContentValues v) { return null; }
    @Override public int delete(Uri u, String s, String[] a) { return 0; }
    @Override public int update(Uri u, ContentValues v, String s, String[] a) { return 0; }
}
