python -m graphphysics.train \
            --project_name=Hybrid_Physics_GNN \
            --training_parameters_path=training_config/cylinder.json \
            --num_epochs=2 \
            --init_lr=0.001 \
            --batch_size=2 \
            --warmup=500 \
            --num_workers=0 \
            --prefetch_factor=0 \
            --model_save_name=model \