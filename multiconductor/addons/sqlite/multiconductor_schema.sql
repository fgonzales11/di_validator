PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS meta (
  key TEXT PRIMARY KEY,
  value TEXT
);

INSERT OR IGNORE INTO meta (key, value) VALUES ('schema_version', '2');

CREATE TABLE IF NOT EXISTS networks (
  network_id TEXT PRIMARY KEY,
  name TEXT,
  description TEXT,
  sn_mva REAL,
  f_hz REAL,
  rho_ohmm REAL,
  version TEXT,
  format_version TEXT,
  user_pf_options TEXT,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP,
  updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS network_meta (
  network_id TEXT NOT NULL,
  key TEXT NOT NULL,
  value TEXT,
  PRIMARY KEY (network_id, key),
  FOREIGN KEY (network_id) REFERENCES networks(network_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS bus (
  network_id TEXT NOT NULL,
  "index" INTEGER NOT NULL,
  phase INTEGER NOT NULL,
  name TEXT,
  vn_kv REAL,
  grounded INTEGER,
  grounding_r_ohm REAL,
  grounding_x_ohm REAL,
  in_service INTEGER,
  type TEXT,
  zone TEXT,
  PRIMARY KEY (network_id, "index", phase),
  FOREIGN KEY (network_id) REFERENCES networks(network_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS ext_grid (
  network_id TEXT NOT NULL,
  "index" INTEGER NOT NULL,
  circuit INTEGER NOT NULL,
  name TEXT,
  bus INTEGER,
  from_phase INTEGER,
  to_phase INTEGER,
  vm_pu REAL,
  va_degree REAL,
  r_ohm REAL,
  x_ohm REAL,
  in_service INTEGER,
  PRIMARY KEY (network_id, "index", circuit),
  FOREIGN KEY (network_id) REFERENCES networks(network_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS ext_grid_sequence (
  network_id TEXT NOT NULL,
  "index" INTEGER NOT NULL,
  sequence INTEGER NOT NULL,
  name TEXT,
  bus INTEGER,
  from_phase INTEGER,
  to_phase INTEGER,
  vm_pu REAL,
  va_degree REAL,
  r_ohm REAL,
  x_ohm REAL,
  in_service INTEGER,
  PRIMARY KEY (network_id, "index", sequence),
  FOREIGN KEY (network_id) REFERENCES networks(network_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS asymmetric_load (
  network_id TEXT NOT NULL,
  "index" INTEGER NOT NULL,
  circuit INTEGER NOT NULL,
  name TEXT,
  bus INTEGER,
  from_phase INTEGER,
  to_phase INTEGER,
  p_mw REAL,
  q_mvar REAL,
  const_z_percent_p REAL,
  const_i_percent_p REAL,
  const_z_percent_q REAL,
  const_i_percent_q REAL,
  sn_mva REAL,
  scaling REAL,
  in_service INTEGER,
  type TEXT,
  PRIMARY KEY (network_id, "index", circuit),
  FOREIGN KEY (network_id) REFERENCES networks(network_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS asymmetric_sgen (
  network_id TEXT NOT NULL,
  "index" INTEGER NOT NULL,
  circuit INTEGER NOT NULL,
  name TEXT,
  bus INTEGER,
  from_phase INTEGER,
  to_phase INTEGER,
  p_mw REAL,
  q_mvar REAL,
  vm_pu REAL,
  const_z_percent_p REAL,
  const_i_percent_p REAL,
  const_z_percent_q REAL,
  const_i_percent_q REAL,
  sn_mva REAL,
  scaling REAL,
  in_service INTEGER,
  type TEXT,
  current_source INTEGER,
  slack INTEGER,
  control_mode TEXT,
  control_curve TEXT,
  PRIMARY KEY (network_id, "index", circuit),
  FOREIGN KEY (network_id) REFERENCES networks(network_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS asymmetric_gen (
  network_id TEXT NOT NULL,
  "index" INTEGER NOT NULL,
  circuit INTEGER NOT NULL,
  name TEXT,
  bus INTEGER,
  from_phase INTEGER,
  to_phase INTEGER,
  p_mw REAL,
  vm_pu REAL,
  sn_mva REAL,
  scaling REAL,
  in_service INTEGER,
  type TEXT,
  current_source INTEGER,
  slack INTEGER,
  PRIMARY KEY (network_id, "index", circuit),
  FOREIGN KEY (network_id) REFERENCES networks(network_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS asymmetric_shunt (
  network_id TEXT NOT NULL,
  "index" INTEGER NOT NULL,
  circuit INTEGER NOT NULL,
  name TEXT,
  bus INTEGER,
  from_phase INTEGER,
  to_phase INTEGER,
  p_mw REAL,
  q_mvar REAL,
  control_mode TEXT,
  closed INTEGER,
  v_threshold_on REAL,
  v_threshold_off REAL,
  max_q_mvar REAL,
  max_p_mw REAL,
  vn_kv REAL,
  in_service INTEGER,
  PRIMARY KEY (network_id, "index", circuit),
  FOREIGN KEY (network_id) REFERENCES networks(network_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS line (
  network_id TEXT NOT NULL,
  "index" INTEGER NOT NULL,
  circuit INTEGER NOT NULL,
  name TEXT,
  std_type TEXT,
  model_type TEXT,
  from_bus INTEGER,
  from_phase INTEGER,
  to_bus INTEGER,
  to_phase INTEGER,
  length_km REAL,
  type TEXT,
  in_service INTEGER,
  PRIMARY KEY (network_id, "index", circuit),
  FOREIGN KEY (network_id) REFERENCES networks(network_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS switch (
  network_id TEXT NOT NULL,
  "index" INTEGER NOT NULL,
  circuit INTEGER NOT NULL,
  bus INTEGER,
  phase INTEGER,
  element INTEGER,
  et TEXT,
  type TEXT,
  closed INTEGER,
  name TEXT,
  r_ohm REAL,
  PRIMARY KEY (network_id, "index", circuit),
  FOREIGN KEY (network_id) REFERENCES networks(network_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS trafo1ph (
  network_id TEXT NOT NULL,
  "index" INTEGER NOT NULL,
  bus INTEGER NOT NULL,
  circuit INTEGER NOT NULL,
  name TEXT,
  from_phase INTEGER,
  to_phase INTEGER,
  vn_kv REAL,
  sn_mva REAL,
  vk_percent REAL,
  vkr_percent REAL,
  pfe_kw REAL,
  i0_percent REAL,
  tap_side TEXT,
  tap_neutral INTEGER,
  tap_min INTEGER,
  tap_max INTEGER,
  tap_step_percent REAL,
  tap_step_degree REAL,
  tap_pos INTEGER,
  in_service INTEGER,
  PRIMARY KEY (network_id, "index", bus, circuit),
  FOREIGN KEY (network_id) REFERENCES networks(network_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS measurement (
  network_id TEXT NOT NULL,
  "index" INTEGER NOT NULL,
  name TEXT,
  measurement_type TEXT,
  element_type TEXT,
  element INTEGER,
  value REAL,
  std_dev REAL,
  side TEXT,
  PRIMARY KEY (network_id, "index"),
  FOREIGN KEY (network_id) REFERENCES networks(network_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS pwl_cost (
  network_id TEXT NOT NULL,
  "index" INTEGER NOT NULL,
  power_type TEXT,
  element INTEGER,
  et TEXT,
  points TEXT,
  PRIMARY KEY (network_id, "index"),
  FOREIGN KEY (network_id) REFERENCES networks(network_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS poly_cost (
  network_id TEXT NOT NULL,
  "index" INTEGER NOT NULL,
  element INTEGER,
  et TEXT,
  cp0_eur REAL,
  cp1_eur_per_mw REAL,
  cp2_eur_per_mw2 REAL,
  cq0_eur REAL,
  cq1_eur_per_mvar REAL,
  cq2_eur_per_mvar2 REAL,
  PRIMARY KEY (network_id, "index"),
  FOREIGN KEY (network_id) REFERENCES networks(network_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS characteristic (
  network_id TEXT NOT NULL,
  "index" INTEGER NOT NULL,
  object TEXT,
  PRIMARY KEY (network_id, "index"),
  FOREIGN KEY (network_id) REFERENCES networks(network_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS controller (
  network_id TEXT NOT NULL,
  "index" INTEGER NOT NULL,
  object TEXT,
  in_service INTEGER,
  "order" REAL,
  level TEXT,
  initial_run INTEGER,
  recycle TEXT,
  PRIMARY KEY (network_id, "index"),
  FOREIGN KEY (network_id) REFERENCES networks(network_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS "group" (
  network_id TEXT NOT NULL,
  "index" INTEGER NOT NULL,
  name TEXT,
  element_type TEXT,
  element TEXT,
  reference_column TEXT,
  PRIMARY KEY (network_id, "index"),
  FOREIGN KEY (network_id) REFERENCES networks(network_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS line_geodata (
  network_id TEXT NOT NULL,
  "index" INTEGER NOT NULL,
  coords TEXT,
  PRIMARY KEY (network_id, "index"),
  FOREIGN KEY (network_id) REFERENCES networks(network_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS bus_geodata (
  network_id TEXT NOT NULL,
  "index" INTEGER NOT NULL,
  x REAL,
  y REAL,
  coords TEXT,
  PRIMARY KEY (network_id, "index"),
  FOREIGN KEY (network_id) REFERENCES networks(network_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS configuration_std_type (
  network_id TEXT NOT NULL,
  "index" INTEGER NOT NULL,
  circuit INTEGER NOT NULL,
  conductor_outer_diameter_m REAL,
  gmr_coefficient REAL,
  r_dc_ohm_per_km REAL,
  g_us_per_km REAL,
  max_i_ka REAL,
  x_m REAL,
  y_m REAL,
  PRIMARY KEY (network_id, "index", circuit),
  FOREIGN KEY (network_id) REFERENCES networks(network_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS sequence_std_type (
  network_id TEXT NOT NULL,
  "index" INTEGER NOT NULL,
  name TEXT,
  r_ohm_per_km REAL,
  x_ohm_per_km REAL,
  r0_ohm_per_km REAL,
  x0_ohm_per_km REAL,
  c_nf_per_km REAL,
  c0_nf_per_km REAL,
  max_i_ka REAL,
  PRIMARY KEY (network_id, "index"),
  FOREIGN KEY (network_id) REFERENCES networks(network_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS matrix_std_type (
  network_id TEXT NOT NULL,
  "index" INTEGER NOT NULL,
  circuit INTEGER NOT NULL,
  name TEXT,
  max_i_ka REAL,
  r_1_ohm_per_km REAL,
  r_2_ohm_per_km REAL,
  r_3_ohm_per_km REAL,
  r_4_ohm_per_km REAL,
  x_1_ohm_per_km REAL,
  x_2_ohm_per_km REAL,
  x_3_ohm_per_km REAL,
  x_4_ohm_per_km REAL,
  g_1_us_per_km REAL,
  g_2_us_per_km REAL,
  g_3_us_per_km REAL,
  g_4_us_per_km REAL,
  b_1_us_per_km REAL,
  b_2_us_per_km REAL,
  b_3_us_per_km REAL,
  b_4_us_per_km REAL,
  PRIMARY KEY (network_id, "index", circuit),
  FOREIGN KEY (network_id) REFERENCES networks(network_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS res_bus (
  network_id TEXT NOT NULL,
  "index" INTEGER NOT NULL,
  phase INTEGER NOT NULL,
  vm_pu REAL,
  va_degree REAL,
  p_mw REAL,
  q_mvar REAL,
  imbalance_percent REAL,
  PRIMARY KEY (network_id, "index", phase),
  FOREIGN KEY (network_id) REFERENCES networks(network_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS res_line (
  network_id TEXT NOT NULL,
  "index" INTEGER NOT NULL,
  circuit INTEGER NOT NULL,
  i_from_ka REAL,
  ia_from_degree REAL,
  i_to_ka REAL,
  ia_to_degree REAL,
  i_ka REAL,
  p_from_mw REAL,
  p_to_mw REAL,
  q_from_mvar REAL,
  q_to_mvar REAL,
  pl_mw REAL,
  ql_mvar REAL,
  vm_from_pu REAL,
  vm_to_pu REAL,
  va_from_degree REAL,
  va_to_degree REAL,
  loading_percent REAL,
  PRIMARY KEY (network_id, "index", circuit),
  FOREIGN KEY (network_id) REFERENCES networks(network_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS res_trafo (
  network_id TEXT NOT NULL,
  "index" INTEGER NOT NULL,
  bus INTEGER NOT NULL,
  circuit INTEGER NOT NULL,
  p_mw REAL,
  q_mvar REAL,
  i_ka REAL,
  vm_pu REAL,
  va_degree REAL,
  pl_mw REAL,
  ql_mvar REAL,
  loading_percent REAL,
  PRIMARY KEY (network_id, "index", bus, circuit),
  FOREIGN KEY (network_id) REFERENCES networks(network_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS res_ext_grid (
  network_id TEXT NOT NULL,
  "index" INTEGER NOT NULL,
  circuit INTEGER NOT NULL,
  p_mw REAL,
  q_mvar REAL,
  PRIMARY KEY (network_id, "index", circuit),
  FOREIGN KEY (network_id) REFERENCES networks(network_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS res_ext_grid_sequence (
  network_id TEXT NOT NULL,
  "index" INTEGER NOT NULL,
  sequence INTEGER NOT NULL,
  p_mw REAL,
  q_mvar REAL,
  PRIMARY KEY (network_id, "index", sequence),
  FOREIGN KEY (network_id) REFERENCES networks(network_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS res_asymmetric_load (
  network_id TEXT NOT NULL,
  "index" INTEGER NOT NULL,
  circuit INTEGER NOT NULL,
  p_mw REAL,
  q_mvar REAL,
  PRIMARY KEY (network_id, "index", circuit),
  FOREIGN KEY (network_id) REFERENCES networks(network_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS res_asymmetric_sgen (
  network_id TEXT NOT NULL,
  "index" INTEGER NOT NULL,
  circuit INTEGER NOT NULL,
  p_mw REAL,
  q_mvar REAL,
  va_degree REAL,
  vm_pu REAL,
  PRIMARY KEY (network_id, "index", circuit),
  FOREIGN KEY (network_id) REFERENCES networks(network_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS res_asymmetric_gen (
  network_id TEXT NOT NULL,
  "index" INTEGER NOT NULL,
  circuit INTEGER NOT NULL,
  p_mw REAL,
  q_mvar REAL,
  va_degree REAL,
  vm_pu REAL,
  PRIMARY KEY (network_id, "index", circuit),
  FOREIGN KEY (network_id) REFERENCES networks(network_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS res_asymmetric_shunt (
  network_id TEXT NOT NULL,
  "index" INTEGER NOT NULL,
  circuit INTEGER NOT NULL,
  p_mw REAL,
  q_mvar REAL,
  PRIMARY KEY (network_id, "index", circuit),
  FOREIGN KEY (network_id) REFERENCES networks(network_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS _empty_res_bus (
  network_id TEXT NOT NULL,
  "index" INTEGER NOT NULL,
  vm_pu REAL,
  va_degree REAL,
  p_mw REAL,
  q_mvar REAL,
  PRIMARY KEY (network_id, "index"),
  FOREIGN KEY (network_id) REFERENCES networks(network_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS _empty_res_line (
  network_id TEXT NOT NULL,
  "index" INTEGER NOT NULL,
  p_from_mw REAL,
  q_from_mvar REAL,
  p_to_mw REAL,
  q_to_mvar REAL,
  pl_mw REAL,
  ql_mvar REAL,
  i_from_ka REAL,
  i_to_ka REAL,
  i_ka REAL,
  vm_from_pu REAL,
  va_from_degree REAL,
  vm_to_pu REAL,
  va_to_degree REAL,
  loading_percent REAL,
  PRIMARY KEY (network_id, "index"),
  FOREIGN KEY (network_id) REFERENCES networks(network_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS _empty_res_trafo (
  network_id TEXT NOT NULL,
  "index" INTEGER NOT NULL,
  p_mw REAL,
  q_mvar REAL,
  pl_mw REAL,
  ql_mvar REAL,
  i_ka REAL,
  vm_pu REAL,
  va_degree REAL,
  loading_percent REAL,
  PRIMARY KEY (network_id, "index"),
  FOREIGN KEY (network_id) REFERENCES networks(network_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS _empty_res_asymmetric_load (
  network_id TEXT NOT NULL,
  "index" INTEGER NOT NULL,
  p_mw REAL,
  q_mvar REAL,
  PRIMARY KEY (network_id, "index"),
  FOREIGN KEY (network_id) REFERENCES networks(network_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS _empty_res_asymmetric_sgen (
  network_id TEXT NOT NULL,
  "index" INTEGER NOT NULL,
  p_mw REAL,
  q_mvar REAL,
  va_degree REAL,
  vm_pu REAL,
  PRIMARY KEY (network_id, "index"),
  FOREIGN KEY (network_id) REFERENCES networks(network_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS _empty_res_asymmetric_gen (
  network_id TEXT NOT NULL,
  "index" INTEGER NOT NULL,
  p_mw REAL,
  q_mvar REAL,
  va_degree REAL,
  vm_pu REAL,
  PRIMARY KEY (network_id, "index"),
  FOREIGN KEY (network_id) REFERENCES networks(network_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS _empty_res_ext_grid (
  network_id TEXT NOT NULL,
  "index" INTEGER NOT NULL,
  p_mw REAL,
  q_mvar REAL,
  PRIMARY KEY (network_id, "index"),
  FOREIGN KEY (network_id) REFERENCES networks(network_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS _empty_res_ext_grid_sequence (
  network_id TEXT NOT NULL,
  "index" INTEGER NOT NULL,
  p_mw REAL,
  q_mvar REAL,
  PRIMARY KEY (network_id, "index"),
  FOREIGN KEY (network_id) REFERENCES networks(network_id) ON DELETE CASCADE
);
