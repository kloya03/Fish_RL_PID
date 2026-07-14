# Policy Application Notes (Deployment)

## Observations used by policy

```
obs = [
    ux,                 # forward velocity
    hd_err,             # wrapped heading error
    delta_prev          # previous servo angle
]
```

Where:

* `hd_err = wrap(qh - heading_desired)`
* wrap implemented as:

```
hd_err = atan2(sin(err), cos(err))
```

---

## Action from policy

Policy outputs:

```
kp_raw ∈ [-1,1]
kd_raw ∈ [-1,1]
```

Convert to actual gains:

```
kp_min, kp_max = 0.0, 5.0
kd_min, kd_max = 0.0, 5.0

kp = kp_min + 0.5*(kp_raw + 1)*(kp_max - kp_min)
kd = kd_min + 0.5*(kd_raw + 1)*(kd_max - kd_min)
```

---

## Heading PD controller

```
hd_error = wrap(qh - heading_desired)

delta = kp * hd_error
      + kd * (hd_error - heading_error_prev)/dt
```

---

## Servo rate limit

```
delta_change = delta - delta_prev
delta_change = clip(delta_change,
                    -delta_rate_max*dt,
                     delta_rate_max*dt)

delta = delta_prev + delta_change
```

---

## Servo limits

```
delta = clip(delta, -delta_max, delta_max)
```

---

delta_max = 1.3 rad
delta_rate = 5rad/s

