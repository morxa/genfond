(define (problem tireworld-052-04)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l22 l23 l24 l25 l26 l27 l28 l29 l3 l30 l31 l32 l33 l34 l35 l36 l37 l38 l39 l4 l40 l41 l42 l43 l44 l45 l46 l47 l48 l49 l5 l50 l51 l6 l7 l8 l9 - location)
    (:init (road l0 l6) (road l1 l21) (road l1 l33) (road l10 l3) (road l11 l2) (road l11 l28) (road l12 l15) (road l12 l31) (road l13 l32) (road l13 l34) (road l14 l12) (road l14 l35) (road l15 l12) (road l15 l42) (road l15 l43) (road l16 l32) (road l16 l48) (road l17 l23) (road l17 l38) (road l18 l20) (road l18 l46) (road l19 l43) (road l19 l45) (road l2 l9) (road l20 l10) (road l20 l18) (road l21 l1) (road l21 l23) (road l21 l27) (road l22 l21) (road l23 l17) (road l23 l21) (road l23 l24) (road l24 l23) (road l24 l31) (road l25 l36) (road l25 l44) (road l26 l36) (road l26 l40) (road l27 l21) (road l27 l8) (road l28 l11) (road l28 l6) (road l29 l33) (road l29 l38) (road l3 l10) (road l3 l49) (road l30 l41) (road l31 l12) (road l31 l24) (road l31 l4) (road l32 l13) (road l33 l1) (road l34 l13) (road l34 l45) (road l35 l14) (road l35 l41) (road l36 l25) (road l36 l26) (road l37 l5) (road l37 l7) (road l38 l17) (road l38 l29) (road l39 l22) (road l4 l0) (road l4 l31) (road l4 l50) (road l40 l26) (road l40 l46) (road l41 l30) (road l41 l35) (road l41 l47) (road l42 l15) (road l43 l15) (road l43 l19) (road l44 l25) (road l45 l19) (road l45 l34) (road l46 l18) (road l46 l40) (road l47 l41) (road l47 l51) (road l48 l16) (road l49 l3) (road l49 l51) (road l5 l37) (road l5 l8) (road l50 l4) (road l50 l44) (road l51 l47) (road l51 l49) (road l6 l0) (road l6 l28) (road l7 l48) (road l8 l27) (road l8 l5) (road l9 l2) (road l9 l30) (spare-in l0) (spare-in l1) (spare-in l10) (spare-in l11) (spare-in l12) (spare-in l14) (spare-in l15) (spare-in l16) (spare-in l17) (spare-in l19) (spare-in l21) (spare-in l22) (spare-in l23) (spare-in l24) (spare-in l25) (spare-in l26) (spare-in l27) (spare-in l28) (spare-in l29) (spare-in l31) (spare-in l33) (spare-in l35) (spare-in l36) (spare-in l37) (spare-in l38) (spare-in l41) (spare-in l43) (spare-in l44) (spare-in l45) (spare-in l46) (spare-in l47) (spare-in l48) (spare-in l5) (spare-in l51) (spare-in l6) (spare-in l7) (spare-in l8) (spare-in l9) (vehicle-at l39))
    (:goal (vehicle-at l42))
)