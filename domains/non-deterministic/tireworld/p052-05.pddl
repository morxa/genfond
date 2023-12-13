(define (problem tireworld-052-05)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l22 l23 l24 l25 l26 l27 l28 l29 l3 l30 l31 l32 l33 l34 l35 l36 l37 l38 l39 l4 l40 l41 l42 l43 l44 l45 l46 l47 l48 l49 l5 l50 l51 l6 l7 l8 l9 - location)
    (:init (road l0 l10) (road l0 l40) (road l1 l14) (road l1 l40) (road l10 l0) (road l10 l26) (road l11 l19) (road l11 l43) (road l12 l29) (road l12 l50) (road l13 l17) (road l13 l28) (road l14 l1) (road l14 l23) (road l15 l16) (road l15 l45) (road l16 l15) (road l16 l7) (road l17 l13) (road l17 l23) (road l18 l24) (road l18 l32) (road l19 l11) (road l19 l38) (road l2 l30) (road l2 l36) (road l20 l43) (road l20 l5) (road l21 l42) (road l21 l7) (road l22 l4) (road l23 l14) (road l23 l17) (road l24 l18) (road l24 l44) (road l25 l33) (road l25 l38) (road l25 l39) (road l26 l10) (road l26 l49) (road l27 l5) (road l27 l8) (road l28 l13) (road l29 l12) (road l29 l9) (road l3 l46) (road l3 l8) (road l30 l2) (road l30 l46) (road l31 l35) (road l31 l45) (road l32 l18) (road l32 l51) (road l33 l25) (road l33 l6) (road l34 l4) (road l34 l47) (road l35 l31) (road l35 l47) (road l36 l2) (road l36 l44) (road l37 l41) (road l37 l48) (road l38 l19) (road l38 l25) (road l39 l25) (road l39 l6) (road l4 l22) (road l4 l34) (road l40 l0) (road l40 l1) (road l41 l37) (road l41 l51) (road l42 l21) (road l42 l48) (road l43 l11) (road l43 l20) (road l44 l24) (road l44 l36) (road l45 l15) (road l45 l31) (road l46 l3) (road l46 l30) (road l47 l34) (road l47 l35) (road l48 l37) (road l48 l42) (road l49 l26) (road l49 l50) (road l5 l20) (road l5 l27) (road l50 l12) (road l50 l49) (road l51 l32) (road l51 l41) (road l6 l33) (road l6 l39) (road l6 l9) (road l7 l16) (road l7 l21) (road l8 l27) (road l8 l3) (road l9 l29) (road l9 l6) (spare-in l0) (spare-in l1) (spare-in l10) (spare-in l11) (spare-in l12) (spare-in l13) (spare-in l14) (spare-in l15) (spare-in l16) (spare-in l17) (spare-in l18) (spare-in l19) (spare-in l2) (spare-in l20) (spare-in l21) (spare-in l23) (spare-in l24) (spare-in l25) (spare-in l26) (spare-in l27) (spare-in l29) (spare-in l3) (spare-in l30) (spare-in l31) (spare-in l32) (spare-in l33) (spare-in l34) (spare-in l35) (spare-in l36) (spare-in l37) (spare-in l38) (spare-in l39) (spare-in l4) (spare-in l40) (spare-in l41) (spare-in l42) (spare-in l43) (spare-in l44) (spare-in l45) (spare-in l46) (spare-in l47) (spare-in l48) (spare-in l49) (spare-in l5) (spare-in l50) (spare-in l51) (spare-in l6) (spare-in l7) (spare-in l8) (spare-in l9) (vehicle-at l28))
    (:goal (vehicle-at l22))
)