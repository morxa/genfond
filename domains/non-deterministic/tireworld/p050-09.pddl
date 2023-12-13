(define (problem tireworld-050-09)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l22 l23 l24 l25 l26 l27 l28 l29 l3 l30 l31 l32 l33 l34 l35 l36 l37 l38 l39 l4 l40 l41 l42 l43 l44 l45 l46 l47 l48 l49 l5 l6 l7 l8 l9 - location)
    (:init (road l0 l33) (road l0 l42) (road l0 l8) (road l1 l10) (road l1 l44) (road l10 l1) (road l10 l9) (road l11 l16) (road l11 l39) (road l11 l49) (road l12 l40) (road l12 l45) (road l13 l41) (road l13 l5) (road l14 l15) (road l14 l33) (road l15 l14) (road l15 l48) (road l16 l11) (road l16 l19) (road l16 l26) (road l17 l21) (road l17 l31) (road l18 l19) (road l18 l32) (road l18 l46) (road l19 l16) (road l19 l18) (road l2 l29) (road l2 l39) (road l20 l29) (road l20 l30) (road l21 l17) (road l21 l30) (road l22 l37) (road l22 l4) (road l23 l32) (road l23 l42) (road l24 l8) (road l25 l44) (road l25 l47) (road l26 l16) (road l26 l4) (road l27 l37) (road l27 l47) (road l28 l41) (road l28 l9) (road l29 l2) (road l29 l20) (road l3 l43) (road l3 l7) (road l30 l20) (road l30 l21) (road l31 l17) (road l31 l42) (road l32 l18) (road l32 l23) (road l33 l0) (road l33 l14) (road l34 l45) (road l34 l6) (road l35 l43) (road l35 l44) (road l36 l38) (road l36 l46) (road l37 l22) (road l37 l27) (road l38 l36) (road l39 l11) (road l39 l2) (road l4 l22) (road l4 l26) (road l40 l12) (road l40 l49) (road l41 l13) (road l41 l28) (road l42 l0) (road l42 l23) (road l42 l31) (road l43 l3) (road l43 l35) (road l44 l1) (road l44 l25) (road l44 l35) (road l44 l48) (road l45 l12) (road l45 l34) (road l46 l18) (road l46 l36) (road l46 l7) (road l47 l25) (road l47 l27) (road l48 l15) (road l48 l44) (road l49 l11) (road l49 l40) (road l5 l13) (road l5 l6) (road l6 l34) (road l6 l5) (road l7 l3) (road l7 l46) (road l8 l0) (road l8 l24) (road l9 l10) (road l9 l28) (spare-in l0) (spare-in l1) (spare-in l10) (spare-in l11) (spare-in l12) (spare-in l15) (spare-in l16) (spare-in l17) (spare-in l18) (spare-in l19) (spare-in l2) (spare-in l20) (spare-in l21) (spare-in l22) (spare-in l23) (spare-in l25) (spare-in l26) (spare-in l27) (spare-in l28) (spare-in l29) (spare-in l3) (spare-in l30) (spare-in l31) (spare-in l32) (spare-in l33) (spare-in l34) (spare-in l35) (spare-in l36) (spare-in l37) (spare-in l39) (spare-in l4) (spare-in l40) (spare-in l41) (spare-in l42) (spare-in l43) (spare-in l44) (spare-in l45) (spare-in l46) (spare-in l47) (spare-in l48) (spare-in l49) (spare-in l5) (spare-in l6) (spare-in l7) (spare-in l8) (spare-in l9) (vehicle-at l38))
    (:goal (vehicle-at l24))
)