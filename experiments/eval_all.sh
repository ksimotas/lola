#!/usr/bin/bash

runs=(
    "/mnt/ceph/users/frozet/lola/runs/dm/otocl205_euler_all_f32c4_vit_large"
    "/mnt/ceph/users/frozet/lola/runs/dm/oy7s3wfq_euler_all_f32c16_vit_large"
    "/mnt/ceph/users/frozet/lola/runs/dm/6mnf6vle_euler_all_f32c32_vit_large"
    "/mnt/ceph/users/frozet/lola/runs/dm/e7xzovde_euler_all_f32c64_vit_large"
    "/mnt/ceph/users/frozet/lola/runs/dm/cyqmbfjb_rayleigh_benard_f32c4_vit_large"
    "/mnt/ceph/users/frozet/lola/runs/dm/667t8kzm_rayleigh_benard_f32c16_vit_large"
    "/mnt/ceph/users/frozet/lola/runs/dm/zt8nzq37_rayleigh_benard_f32c32_vit_large"
    "/mnt/ceph/users/frozet/lola/runs/dm/0fqjt3js_rayleigh_benard_f32c64_vit_large"
    "/mnt/ceph/users/rohana/lola/runs/sm/a3hc2as6_euler_all_f32c4_vit_large"
    "/mnt/ceph/users/frozet/lola/runs/sm/95zfkk6w_euler_all_f32c16_vit_large"
    "/mnt/ceph/users/frozet/lola/runs/sm/9a2wk0hd_euler_all_f32c32_vit_large"
    "/mnt/ceph/users/frozet/lola/runs/sm/s259b4l6_euler_all_f32c64_vit_large"
    "/mnt/ceph/users/rohana/lola/runs/sm/chua0k42_rayleigh_benard_f32c4_vit_large"
    "/mnt/ceph/users/frozet/lola/runs/sm/xp1prkdo_rayleigh_benard_f32c16_vit_large"
    "/mnt/ceph/users/frozet/lola/runs/sm/fa6jeugp_rayleigh_benard_f32c32_vit_large"
    "/mnt/ceph/users/frozet/lola/runs/sm/lrg1qgi2_rayleigh_benard_f32c64_vit_large"
    "/mnt/ceph/users/frozet/lola/runs/sm/1y8qrgee_euler_all_pixel_vit_pixel"
    "/mnt/ceph/users/frozet/lola/runs/sm/ses7lyf5_rayleigh_benard_pixel_vit_pixel"
)

for run in "${runs[@]}"
do
    if [[ $run = *"rayleigh_benard"* ]]
    then
        starts=(4 16 64)
    else
        starts=(0)
    fi

    for start in "${starts[@]}"
    do
        for context in $(seq 1 3)
        do
            python eval.py run=$run start=$start context=$context overlap=$context
        done
    done
done
