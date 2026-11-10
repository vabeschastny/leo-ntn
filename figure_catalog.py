"""Explicit allowlist linked to page numbers of the supervisor's 51-page PDF."""
FIGURES={
 1:('core','01_failure_composition'),2:('core','02_access_success'),3:('core','03_completion_combined'),
 8:('core','04_delay_mean'),12:('core','07_energy_per_success'),13:('core','08_energy_until_success_mean'),18:('core','energy_unsuccessful'),
 19:('threshold_wait','03_capped_wait'),20:('channel_map','individual_channels_100'),
 21:('mapping','delay_s_mean'),23:('mapping','energy_per_success_j'),24:('mapping','payload_collision_probability'),26:('mapping','success_ratio'),
 27:('repetitions','delay_s_mean'),29:('repetitions','energy_per_success_j'),32:('repetitions','success_ratio'),
 33:('trigger','01_failure_vs_trigger_elevation'),34:('trigger','delay_s_mean'),36:('trigger','energy_per_success_j'),
 40:('phy','delay_s_mean'),42:('phy','energy_per_success_j'),43:('phy','payload_collision_probability'),44:('phy','preamble_collision_probability'),48:('power','energy_per_success_j')}
MERGED={4:3,5:3,6:3,7:3}
REMOVED=set(range(1,52))-set(FIGURES)-set(MERGED)
def path_for(root,number):
    suite,stem=FIGURES[number]
    return root/suite/(stem+'.png') if suite in ('threshold_wait','channel_map') else root/suite/'figures'/(stem+'.png')
