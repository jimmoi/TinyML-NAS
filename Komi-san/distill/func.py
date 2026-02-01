import tensorflow as tf
import numpy as np

def save_teacher_knowledge(teacher_model, dataset, save_path):
    all_logits = []
    all_images = []
    all_labels = []

    print("Extracting teacher knowledge...")
    for images, labels in dataset:
        # Get logits (predictions BEFORE softmax is better for distillation)
        # We can extract them by taking the output of the last dense layer
        logits = teacher_model.predict(images)
        
        all_images.append(images.numpy())
        all_labels.append(labels.numpy())
        all_logits.append(logits)

    # Concatenate everything into numpy arrays
    np.savez_compressed(save_path, 
                        x=np.concatenate(all_images), 
                        y=np.concatenate(all_labels), 
                        logits=np.concatenate(all_logits))
    print(f"Knowledge saved to {save_path}")

def load_distillation_data(file_path, batch_size=32):
    data = np.load(file_path)
    
    # Create a dataset that yields (Images, [Hard Labels, Teacher Logits])
    dataset = tf.data.Dataset.from_tensor_slices((
        data['x'], 
        {"hard_labels": data['y'], "teacher_logits": data['logits']}
    ))
    
    return dataset.shuffle(1000).batch(batch_size)

def distillation_loss(y_true_dict, y_pred_student, temperature=3, alpha=0.1):
    y_true = y_true_dict["hard_labels"]
    teacher_logits = y_true_dict["teacher_logits"]
    
    # 1. Standard Cross-Entropy (Student vs Ground Truth)
    student_loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred_student)
    
    # 2. Distillation Loss (Student vs Teacher Soft Labels)
    # We apply softmax with temperature to both
    soft_teacher = tf.nn.softmax(teacher_logits / temperature)
    soft_student = tf.nn.softmax(y_pred_student / temperature) # Note: Student output should be logits
    
    distill_loss = tf.keras.losses.kl_divergence(soft_teacher, soft_student)
    
    return alpha * student_loss + (1 - alpha) * (temperature**2) * distill_loss

# class Distiller(tf.keras.Model):
    def __init__(self, student, teacher):
        super().__init__()
        self.teacher = teacher
        self.student = student

    def compile(self, optimizer, metrics, student_loss_fn, distillation_loss_fn, alpha=0.1, temperature=3):
        super().compile(optimizer=optimizer, metrics=metrics)
        self.student_loss_fn = student_loss_fn
        self.distillation_loss_fn = distillation_loss_fn
        self.alpha = alpha # Weight of student loss
        self.temperature = temperature

    def train_step(self, data):
        x, y = data
        
        # Teacher must be in inference mode
        teacher_predictions = self.teacher(x, training=False)

        with tf.GradientTape() as tape:
            student_predictions = self.student(x, training=True)

            # Standard loss
            student_loss = self.student_loss_fn(y, student_predictions)
            
            # Distillation loss (KL Divergence on Soft Labels)
            distillation_loss = self.distillation_loss_fn(
                tf.nn.softmax(teacher_predictions / self.temperature, axis=1),
                tf.nn.softmax(student_predictions / self.temperature, axis=1),
            )

            loss = self.alpha * student_loss + (1 - self.alpha) * distillation_loss

        trainable_vars = self.student.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        self.compiled_metrics.update_state(y, student_predictions)
        results = {m.name: m.result() for m in self.metrics}
        results.update({"loss": loss, "distillation_loss": distillation_loss})
        return results