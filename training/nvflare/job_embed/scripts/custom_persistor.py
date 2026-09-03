import os
import shutil

from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.fl_context import FLContext
from nvflare.app_opt.pt.file_model_persistor import PTFileModelPersistor


class CustomPersistor(PTFileModelPersistor):
    def save_model(self, ml, fl_ctx: FLContext):
        try:
            self.log_info(fl_ctx, "Saving model using CustomPersistor")
            super().save_model(ml, fl_ctx)
        except Exception as e:
            self.log_error(fl_ctx, f"Super().save_model failed: {e}")

        try:
            source_file = getattr(self, "_ckpt_save_path", None)
            if not source_file:
                app_root = fl_ctx.get_prop(FLContextKey.APP_ROOT).__str__()
                source_file = os.path.join(app_root, "fl_model.pt")

            target_dir = "/app/out"
            target_file = os.path.join(target_dir, "GLOBAL_MODEL.pt")

            if source_file and os.path.exists(source_file):
                shutil.copy2(source_file, target_file)
                size_mb = os.path.getsize(target_file) / 1024 / 1024
                self.log_info(fl_ctx, f"Model saved: {target_file} ({size_mb:.2f} MB)")
            else:
                self.log_warning(fl_ctx, f"Source model not found: {source_file}")
        except Exception as e:
            self.log_error(fl_ctx, f"Copy failed: {e}")
