(define (problem tireworld-052-07)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l22 l23 l24 l25 l26 l27 l28 l29 l3 l30 l31 l32 l33 l34 l35 l36 l37 l38 l39 l4 l40 l41 l42 l43 l44 l45 l46 l47 l48 l49 l5 l50 l51 l6 l7 l8 l9 - location)
    (:init (road l0 l12) (road l0 l23) (road l1 l24) (road l1 l33) (road l10 l28) (road l10 l41) (road l11 l14) (road l11 l17) (road l12 l0) (road l13 l20) (road l13 l27) (road l14 l11) (road l15 l19) (road l15 l41) (road l16 l40) (road l17 l20) (road l18 l27) (road l18 l46) (road l19 l50) (road l2 l20) (road l2 l35) (road l2 l49) (road l20 l13) (road l20 l2) (road l21 l36) (road l22 l31) (road l22 l51) (road l23 l0) (road l23 l5) (road l24 l1) (road l24 l38) (road l25 l12) (road l25 l37) (road l26 l51) (road l27 l13) (road l27 l18) (road l27 l48) (road l28 l10) (road l29 l43) (road l3 l42) (road l3 l47) (road l30 l16) (road l30 l49) (road l30 l8) (road l31 l22) (road l31 l8) (road l32 l40) (road l33 l1) (road l33 l21) (road l34 l14) (road l35 l2) (road l35 l42) (road l36 l21) (road l36 l39) (road l37 l25) (road l37 l45) (road l38 l24) (road l38 l5) (road l39 l47) (road l4 l9) (road l40 l16) (road l40 l32) (road l40 l7) (road l41 l10) (road l41 l15) (road l42 l3) (road l42 l35) (road l43 l29) (road l43 l44) (road l44 l32) (road l44 l43) (road l45 l37) (road l46 l18) (road l46 l45) (road l47 l3) (road l47 l39) (road l48 l27) (road l48 l4) (road l49 l2) (road l49 l30) (road l5 l38) (road l50 l19) (road l51 l22) (road l6 l26) (road l7 l28) (road l8 l30) (road l8 l31) (road l9 l29) (road l9 l4) (road l9 l6) (spare-in l0) (spare-in l1) (spare-in l10) (spare-in l11) (spare-in l12) (spare-in l13) (spare-in l14) (spare-in l15) (spare-in l17) (spare-in l19) (spare-in l20) (spare-in l22) (spare-in l23) (spare-in l24) (spare-in l26) (spare-in l27) (spare-in l28) (spare-in l29) (spare-in l30) (spare-in l32) (spare-in l37) (spare-in l39) (spare-in l4) (spare-in l40) (spare-in l41) (spare-in l42) (spare-in l43) (spare-in l44) (spare-in l48) (spare-in l5) (spare-in l51) (spare-in l7) (spare-in l9) (vehicle-at l34))
    (:goal (vehicle-at l50))
)