package com.hexanima.npu;

import android.app.Notification;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.Service;
import android.content.Intent;
import android.os.IBinder;

/** Only keeps the process alive (foreground service, type dataSync) while model files are downloaded; the work itself runs in Main. */
public class DownloadService extends Service {
    @Override public int onStartCommand(Intent i, int f, int id) {
        L.init(this);
        NotificationManager nm = getSystemService(NotificationManager.class);
        boolean stop = i != null && "stop".equals(i.getStringExtra("mode")), gen = i != null && "gen".equals(i.getStringExtra("mode"));   // keeps the process alive while a picture is generated with the app in the background
        nm.createNotificationChannel(new NotificationChannel(gen ? "gen" : "dl", gen ? L.s("Генерация", "Generation") : L.s("Загрузка моделей", "Model downloads"), NotificationManager.IMPORTANCE_LOW));
        Notification n = new Notification.Builder(this, gen ? "gen" : "dl").setSmallIcon(gen ? android.R.drawable.ic_menu_gallery : android.R.drawable.stat_sys_download).setContentTitle("HexAnima")
            .setContentText(gen ? L.s("Генерирую картинку…", "Generating a picture…") : L.s("Скачиваю файлы модели…", "Downloading model files…")).setOngoing(true).build();
        startForeground(1, n);   // always: every startForegroundService must be followed by it, or Android kills the app
        if (stop) { stopForeground(STOP_FOREGROUND_REMOVE); stopSelf(); }
        return START_NOT_STICKY;
    }
    /** stopService right after startForegroundService (a generation that fails at once) crashes the app: stop through the service itself, the intents are handled in order */
    static void stop(android.content.Context app) { app.startForegroundService(new Intent(app, DownloadService.class).putExtra("mode", "stop")); }
    @Override public IBinder onBind(Intent i) { return null; }
}
