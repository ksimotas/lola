#!/usr/bin/bash

runs=(
    # AEs
    "$HOME/ceph/lola/runs/ae/z522jsw5_euler_all_dcae_f32c4_large"
    "$HOME/ceph/lola/runs/ae/8f21pxzs_euler_all_dcae_f32c16_large"
    "$HOME/ceph/lola/runs/ae/n6g8kix8_euler_all_dcae_f32c64_large"
    "$HOME/ceph/lola/runs/ae/2bkdxjib_rayleigh_benard_dcae_f32c4_large"
    "$HOME/ceph/lola/runs/ae/84hwskd9_rayleigh_benard_dcae_f32c16_large"
    "$HOME/ceph/lola/runs/ae/di2j3rpb_rayleigh_benard_dcae_f32c64_large"
    "$HOME/ceph/lola/runs/ae/gf09iy5g_gravity_cooling_dcae_3d_f8c4_large"
    "$HOME/ceph/lola/runs/ae/0zc7afa4_gravity_cooling_dcae_3d_f8c16_large"
    "$HOME/ceph/lola/runs/ae/dnzp6wv7_gravity_cooling_dcae_3d_f8c64_large"
    # DMs
    "$HOME/ceph/lola/runs/dm/otocl205_euler_all_f32c4_vit_large"
    "$HOME/ceph/lola/runs/dm/oy7s3wfq_euler_all_f32c16_vit_large"
    "$HOME/ceph/lola/runs/dm/e7xzovde_euler_all_f32c64_vit_large"
    "$HOME/ceph/lola/runs/dm/cyqmbfjb_rayleigh_benard_f32c4_vit_large"
    "$HOME/ceph/lola/runs/dm/667t8kzm_rayleigh_benard_f32c16_vit_large"
    "$HOME/ceph/lola/runs/dm/0fqjt3js_rayleigh_benard_f32c64_vit_large"
    "$HOME/ceph/lola/runs/dm/6xekmygr_gravity_cooling_f8c4_vit_large"
    "$HOME/ceph/lola/runs/dm/2cxh0m26_gravity_cooling_f8c16_vit_large"
    "$HOME/ceph/lola/runs/dm/5cywlsx8_gravity_cooling_f8c64_vit_large"
    # SMs
    "$HOME/ceph/lola/runs/sm/a3hc2as6_euler_all_f32c4_vit_large"
    "$HOME/ceph/lola/runs/sm/95zfkk6w_euler_all_f32c16_vit_large"
    "$HOME/ceph/lola/runs/sm/s259b4l6_euler_all_f32c64_vit_large"
    "$HOME/ceph/lola/runs/sm/chua0k42_rayleigh_benard_f32c4_vit_large"
    "$HOME/ceph/lola/runs/sm/xp1prkdo_rayleigh_benard_f32c16_vit_large"
    "$HOME/ceph/lola/runs/sm/lrg1qgi2_rayleigh_benard_f32c64_vit_large"
    "$HOME/ceph/lola/runs/sm/5haia87a_gravity_cooling_f8c4_vit_large"
    "$HOME/ceph/lola/runs/sm/5rtbuoht_gravity_cooling_f8c16_vit_large"
    "$HOME/ceph/lola/runs/sm/e21q2faa_gravity_cooling_f8c64_vit_large"
    "$HOME/ceph/lola/runs/sm/1y8qrgee_euler_all_pixel_vit_pixel"
    "$HOME/ceph/lola/runs/sm/ses7lyf5_rayleigh_benard_pixel_vit_pixel"
)

for run in "${runs[@]}"
do
    if [[ $run = *"rayleigh_benard"* ]]; then
        starts=(8 16)
    elif [[ $run = *"gravity_cooling"* ]]; then
        starts=(4)
    else
        starts=(0)
    fi

    for start in "${starts[@]}"; do
        for context in $(seq 1 3); do
            python eval.py run=$run start=$start context=$context overlap=$context
        done

        python eval.py run=$run start=$start destination="videos" array=7 samples=3 record=3 seed=42
    done
done
