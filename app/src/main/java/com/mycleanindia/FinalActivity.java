package com.mycleanindia;

import android.content.Intent;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Bundle;
import android.support.v7.app.AppCompatActivity;
import android.view.View;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.TextView;

import testing.gps_location.R;

public class FinalActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_final);

        Bundle bundle = getIntent().getBundleExtra("bundle");
        Bitmap bitmap = (Bitmap) bundle.get("bitmap");
        Uri uri = (Uri) bundle.get("uri");
        ImageView iv = (ImageView) findViewById(R.id.imageView2);
        if(uri == null) {
            iv.setImageBitmap(bitmap);
        }else{
            iv.setImageURI(uri);
        }


        EditText editText = (EditText) findViewById(R.id.editText);
        String name = editText.getText().toString();

        TextView address = (TextView) findViewById(R.id.address);
        address.setText(bundle.getString("info"));

        TextView dist = (TextView) findViewById(R.id.dist);
        dist.setText(bundle.getString("dist")+" Municipal Corporation");

    }

    public void onClick3(View v) {
        Intent myIntent = new Intent(this, SubmitActivity.class);
        startActivity(myIntent);
    }
}
