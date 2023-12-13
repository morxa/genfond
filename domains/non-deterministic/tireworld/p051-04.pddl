(define (problem tireworld-051-04)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l22 l23 l24 l25 l26 l27 l28 l29 l3 l30 l31 l32 l33 l34 l35 l36 l37 l38 l39 l4 l40 l41 l42 l43 l44 l45 l46 l47 l48 l49 l5 l50 l6 l7 l8 l9 - location)
    (:init (road l0 l36) (road l0 l38) (road l1 l2) (road l1 l49) (road l10 l13) (road l10 l30) (road l11 l24) (road l11 l45) (road l12 l27) (road l12 l31) (road l12 l47) (road l13 l10) (road l13 l9) (road l14 l3) (road l14 l8) (road l15 l4) (road l15 l42) (road l16 l32) (road l16 l41) (road l17 l22) (road l17 l45) (road l18 l29) (road l18 l34) (road l19 l25) (road l19 l44) (road l2 l1) (road l2 l7) (road l20 l23) (road l20 l39) (road l21 l26) (road l21 l47) (road l22 l17) (road l22 l8) (road l23 l20) (road l23 l9) (road l24 l11) (road l24 l5) (road l25 l19) (road l25 l47) (road l26 l21) (road l26 l33) (road l27 l12) (road l27 l37) (road l28 l35) (road l28 l5) (road l29 l18) (road l29 l50) (road l3 l14) (road l3 l32) (road l30 l10) (road l30 l38) (road l31 l12) (road l31 l44) (road l32 l16) (road l32 l3) (road l33 l26) (road l33 l7) (road l34 l18) (road l34 l38) (road l35 l28) (road l35 l6) (road l36 l0) (road l36 l46) (road l37 l27) (road l37 l43) (road l38 l0) (road l38 l30) (road l38 l34) (road l39 l20) (road l39 l42) (road l4 l15) (road l4 l40) (road l40 l4) (road l40 l49) (road l41 l16) (road l41 l43) (road l42 l15) (road l42 l39) (road l43 l37) (road l43 l41) (road l44 l19) (road l44 l31) (road l45 l11) (road l45 l17) (road l46 l36) (road l46 l48) (road l47 l12) (road l47 l21) (road l47 l25) (road l47 l50) (road l48 l46) (road l48 l6) (road l49 l1) (road l49 l40) (road l5 l24) (road l5 l28) (road l50 l29) (road l50 l47) (road l6 l35) (road l6 l48) (road l7 l2) (road l7 l33) (road l8 l14) (road l8 l22) (road l9 l13) (road l9 l23) (spare-in l0) (spare-in l1) (spare-in l14) (spare-in l18) (spare-in l21) (spare-in l23) (spare-in l25) (spare-in l26) (spare-in l29) (spare-in l30) (spare-in l32) (spare-in l34) (spare-in l42) (spare-in l45) (spare-in l47) (spare-in l5) (spare-in l50) (vehicle-at l19))
    (:goal (vehicle-at l38))
)