(define (problem tireworld-052-03)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l22 l23 l24 l25 l26 l27 l28 l29 l3 l30 l31 l32 l33 l34 l35 l36 l37 l38 l39 l4 l40 l41 l42 l43 l44 l45 l46 l47 l48 l49 l5 l50 l51 l6 l7 l8 l9 - location)
    (:init (road l0 l21) (road l0 l48) (road l1 l10) (road l1 l11) (road l10 l1) (road l10 l13) (road l11 l1) (road l11 l42) (road l12 l15) (road l12 l36) (road l13 l10) (road l13 l32) (road l14 l38) (road l14 l4) (road l14 l6) (road l15 l12) (road l15 l31) (road l16 l46) (road l16 l9) (road l17 l23) (road l17 l7) (road l18 l19) (road l18 l43) (road l19 l18) (road l19 l34) (road l2 l27) (road l2 l31) (road l2 l39) (road l20 l35) (road l20 l47) (road l21 l0) (road l21 l51) (road l22 l24) (road l22 l3) (road l23 l17) (road l23 l45) (road l24 l22) (road l24 l27) (road l24 l51) (road l25 l26) (road l25 l30) (road l25 l40) (road l26 l25) (road l26 l28) (road l27 l2) (road l27 l24) (road l28 l26) (road l28 l44) (road l29 l32) (road l29 l41) (road l3 l22) (road l3 l50) (road l3 l6) (road l30 l25) (road l30 l7) (road l31 l15) (road l31 l2) (road l31 l40) (road l31 l41) (road l32 l13) (road l32 l29) (road l32 l37) (road l33 l5) (road l33 l50) (road l34 l19) (road l34 l4) (road l35 l20) (road l35 l49) (road l36 l12) (road l36 l45) (road l37 l32) (road l37 l49) (road l38 l14) (road l38 l5) (road l39 l2) (road l39 l47) (road l4 l14) (road l4 l34) (road l40 l25) (road l40 l31) (road l41 l29) (road l41 l31) (road l42 l11) (road l42 l8) (road l43 l18) (road l44 l28) (road l44 l46) (road l45 l23) (road l45 l36) (road l46 l16) (road l46 l44) (road l47 l20) (road l47 l39) (road l48 l0) (road l48 l8) (road l49 l35) (road l49 l37) (road l5 l33) (road l5 l38) (road l50 l3) (road l50 l33) (road l51 l21) (road l51 l24) (road l6 l14) (road l6 l3) (road l7 l17) (road l7 l30) (road l8 l42) (road l8 l48) (road l9 l16) (spare-in l0) (spare-in l1) (spare-in l10) (spare-in l11) (spare-in l12) (spare-in l13) (spare-in l14) (spare-in l15) (spare-in l16) (spare-in l17) (spare-in l18) (spare-in l19) (spare-in l2) (spare-in l20) (spare-in l21) (spare-in l22) (spare-in l23) (spare-in l24) (spare-in l25) (spare-in l26) (spare-in l27) (spare-in l28) (spare-in l29) (spare-in l3) (spare-in l30) (spare-in l31) (spare-in l32) (spare-in l33) (spare-in l34) (spare-in l35) (spare-in l36) (spare-in l37) (spare-in l38) (spare-in l39) (spare-in l4) (spare-in l40) (spare-in l41) (spare-in l42) (spare-in l44) (spare-in l45) (spare-in l46) (spare-in l48) (spare-in l49) (spare-in l5) (spare-in l50) (spare-in l51) (spare-in l6) (spare-in l7) (spare-in l8) (vehicle-at l9))
    (:goal (vehicle-at l43))
)