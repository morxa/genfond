(define (problem tireworld-047-05)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l22 l23 l24 l25 l26 l27 l28 l29 l3 l30 l31 l32 l33 l34 l35 l36 l37 l38 l39 l4 l40 l41 l42 l43 l44 l45 l46 l5 l6 l7 l8 l9 - location)
    (:init (road l0 l3) (road l0 l43) (road l1 l26) (road l1 l32) (road l10 l13) (road l10 l29) (road l11 l23) (road l11 l3) (road l12 l24) (road l12 l30) (road l13 l10) (road l13 l22) (road l14 l4) (road l14 l5) (road l15 l33) (road l15 l41) (road l16 l4) (road l17 l32) (road l17 l34) (road l18 l7) (road l18 l8) (road l19 l27) (road l19 l31) (road l2 l22) (road l2 l42) (road l20 l3) (road l20 l37) (road l21 l29) (road l21 l43) (road l22 l13) (road l22 l2) (road l23 l11) (road l23 l33) (road l24 l12) (road l24 l28) (road l25 l38) (road l25 l40) (road l26 l1) (road l26 l44) (road l27 l19) (road l27 l6) (road l28 l24) (road l28 l39) (road l29 l10) (road l29 l21) (road l3 l0) (road l3 l11) (road l3 l20) (road l30 l12) (road l30 l40) (road l31 l19) (road l31 l42) (road l32 l1) (road l32 l17) (road l33 l15) (road l33 l23) (road l34 l17) (road l34 l41) (road l35 l39) (road l35 l8) (road l36 l38) (road l36 l46) (road l37 l20) (road l37 l5) (road l38 l25) (road l38 l36) (road l39 l28) (road l39 l35) (road l4 l14) (road l4 l16) (road l40 l25) (road l40 l30) (road l41 l15) (road l41 l34) (road l42 l2) (road l42 l31) (road l43 l0) (road l43 l21) (road l43 l7) (road l44 l26) (road l44 l45) (road l45 l44) (road l45 l46) (road l46 l36) (road l46 l45) (road l5 l14) (road l5 l37) (road l6 l27) (road l6 l9) (road l7 l18) (road l7 l43) (road l8 l18) (road l8 l35) (road l9 l6) (spare-in l0) (spare-in l1) (spare-in l10) (spare-in l11) (spare-in l12) (spare-in l13) (spare-in l14) (spare-in l15) (spare-in l17) (spare-in l18) (spare-in l19) (spare-in l2) (spare-in l20) (spare-in l21) (spare-in l22) (spare-in l23) (spare-in l24) (spare-in l25) (spare-in l26) (spare-in l27) (spare-in l28) (spare-in l29) (spare-in l3) (spare-in l30) (spare-in l31) (spare-in l32) (spare-in l33) (spare-in l34) (spare-in l35) (spare-in l36) (spare-in l37) (spare-in l38) (spare-in l39) (spare-in l4) (spare-in l40) (spare-in l41) (spare-in l42) (spare-in l43) (spare-in l44) (spare-in l45) (spare-in l46) (spare-in l5) (spare-in l6) (spare-in l7) (spare-in l8) (vehicle-at l9))
    (:goal (vehicle-at l16))
)