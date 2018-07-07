package com.mycleanindia;

import android.content.Intent;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.support.v7.app.AppCompatActivity;
import android.view.View;

import testing.gps_location.R;

public class StartActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_start);
    }


    public void onClick(View v) {
        Intent myIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
        startActivityForResult(myIntent,0);
    }

    public void onClick2(View v) {
        Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT);
        intent.addCategory(Intent.CATEGORY_OPENABLE);
        intent.setType("image/*");
        startActivityForResult(intent, 1);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if(requestCode == 0 && resultCode == RESULT_OK) {
           Bitmap bitmap = (Bitmap) data.getExtras().get("data");
            Intent myIntent = new Intent(this, MainActivity.class);
            myIntent.putExtra("data",bitmap);
            myIntent.putExtra("option",0);
            startActivity(myIntent);
        } else if(requestCode == 1 && resultCode == RESULT_OK) {
            Uri uri = null;
            if (data != null) {
                uri = data.getData();
                Intent myIntent = new Intent(this, MainActivity.class);
                myIntent.putExtra("data",uri);
                myIntent.putExtra("option",1);
                startActivity(myIntent);
            }
        }
    }
}
