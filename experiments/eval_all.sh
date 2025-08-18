#!/usr/bin/bash

runs=(
    # AEs
    "$HOME/ceph/lola/runs/ae/z522jsw5_euler_all_dcae_f32c4_large"
    "$HOME/ceph/lola/runs/ae/8f21pxzs_euler_all_dcae_f32c16_large"
    "$HOME/ceph/lola/runs/ae/n6g8kix8_euler_all_dcae_f32c64_large"
    "$HOME/ceph/lola/runs/ae/8brbetky_rayleigh_benard_dcae_f32c4_large"
    "$HOME/ceph/lola/runs/ae/jiof10wl_rayleigh_benard_dcae_f32c16_large"
    "$HOME/ceph/lola/runs/ae/1e3z5x2c_rayleigh_benard_dcae_f32c64_large"
    "$HOME/ceph/lola/runs/ae/gf09iy5g_gravity_cooling_dcae_3d_f8c4_large"
    "$HOME/ceph/lola/runs/ae/0zc7afa4_gravity_cooling_dcae_3d_f8c16_large"
    "$HOME/ceph/lola/runs/ae/dnzp6wv7_gravity_cooling_dcae_3d_f8c64_large"
    # DMs
    "$HOME/ceph/lola/runs/dm/otocl205_euler_all_f32c4_vit_large"
    "$HOME/ceph/lola/runs/dm/oy7s3wfq_euler_all_f32c16_vit_large"
    "$HOME/ceph/lola/runs/dm/e7xzovde_euler_all_f32c64_vit_large"
    "$HOME/ceph/lola/runs/dm/6hsjhmvw_rayleigh_benard_f32c4_vit_large"
    "$HOME/ceph/lola/runs/dm/e0sdjqy9_rayleigh_benard_f32c16_vit_large"
    "$HOME/ceph/lola/runs/dm/ny04m1tl_rayleigh_benard_f32c64_vit_large"
    "$HOME/ceph/lola/runs/dm/6xekmygr_gravity_cooling_f8c4_vit_large"
    "$HOME/ceph/lola/runs/dm/2cxh0m26_gravity_cooling_f8c16_vit_large"
    "$HOME/ceph/lola/runs/dm/5cywlsx8_gravity_cooling_f8c64_vit_large"
    # SMs
    "$HOME/ceph/lola/runs/sm/a3hc2as6_euler_all_f32c4_vit_large"
    "$HOME/ceph/lola/runs/sm/95zfkk6w_euler_all_f32c16_vit_large"
    "$HOME/ceph/lola/runs/sm/s259b4l6_euler_all_f32c64_vit_large"
    "$HOME/ceph/lola/runs/sm/xqm575ny_rayleigh_benard_f32c4_vit_large"
    "$HOME/ceph/lola/runs/sm/yubzc3d1_rayleigh_benard_f32c16_vit_large"
    "$HOME/ceph/lola/runs/sm/2k83f6km_rayleigh_benard_f32c64_vit_large"
    "$HOME/ceph/lola/runs/sm/5haia87a_gravity_cooling_f8c4_vit_large"
    "$HOME/ceph/lola/runs/sm/5rtbuoht_gravity_cooling_f8c16_vit_large"
    "$HOME/ceph/lola/runs/sm/e21q2faa_gravity_cooling_f8c64_vit_large"
    "$HOME/ceph/lola/runs/sm/1y8qrgee_euler_all_pixel_vit_pixel"
    "$HOME/ceph/lola/runs/sm/43iu4768_rayleigh_benard_pixel_vit_pixel"
)

for run in "${runs[@]}"
do
    if [[ $run = *"rayleigh_benard"* ]]; then
        starts=(16)
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
