package fit.magic.cv

import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import androidx.activity.viewModels
import fit.magic.cv.databinding.ActivityMainBinding
import com.chaquo.python.Python
import com.chaquo.python.android.AndroidPlatform

class MainActivity : AppCompatActivity() {
    private lateinit var activityMainBinding: ActivityMainBinding
    private val viewModel : MainViewModel by viewModels()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        activityMainBinding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(activityMainBinding.root)

        if (! Python.isStarted()) {
            Python.start(AndroidPlatform(this))
        }
    }

    override fun onBackPressed() {
        super.onBackPressed()
        finish()
    }
}
