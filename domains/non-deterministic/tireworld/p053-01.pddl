(define (problem tireworld-053-01)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l22 l23 l24 l25 l26 l27 l28 l29 l3 l30 l31 l32 l33 l34 l35 l36 l37 l38 l39 l4 l40 l41 l42 l43 l44 l45 l46 l47 l48 l49 l5 l50 l51 l52 l6 l7 l8 l9 - location)
    (:init (road l0 l40) (road l0 l49) (road l1 l43) (road l1 l51) (road l10 l21) (road l10 l38) (road l11 l44) (road l11 l5) (road l12 l19) (road l12 l48) (road l13 l29) (road l13 l46) (road l14 l2) (road l14 l32) (road l15 l19) (road l15 l44) (road l16 l28) (road l16 l52) (road l17 l34) (road l17 l37) (road l18 l31) (road l18 l48) (road l19 l12) (road l19 l15) (road l2 l14) (road l2 l8) (road l20 l26) (road l20 l27) (road l21 l10) (road l21 l39) (road l22 l31) (road l22 l33) (road l23 l46) (road l23 l8) (road l24 l45) (road l24 l9) (road l25 l29) (road l25 l47) (road l25 l6) (road l26 l20) (road l26 l35) (road l27 l20) (road l27 l46) (road l28 l16) (road l28 l32) (road l29 l13) (road l29 l25) (road l29 l42) (road l3 l30) (road l3 l4) (road l30 l3) (road l30 l41) (road l30 l52) (road l31 l18) (road l31 l22) (road l32 l14) (road l32 l28) (road l33 l22) (road l33 l39) (road l34 l17) (road l34 l52) (road l35 l26) (road l35 l7) (road l36 l5) (road l36 l9) (road l37 l17) (road l37 l6) (road l38 l10) (road l38 l43) (road l39 l21) (road l39 l33) (road l39 l41) (road l39 l8) (road l4 l3) (road l4 l5) (road l40 l0) (road l40 l45) (road l41 l30) (road l41 l39) (road l42 l29) (road l42 l50) (road l43 l1) (road l43 l38) (road l44 l11) (road l44 l15) (road l45 l24) (road l45 l40) (road l46 l13) (road l46 l23) (road l46 l27) (road l47 l25) (road l47 l51) (road l48 l12) (road l48 l18) (road l49 l0) (road l49 l7) (road l5 l11) (road l5 l36) (road l5 l4) (road l50 l42) (road l51 l1) (road l51 l47) (road l52 l16) (road l52 l30) (road l52 l34) (road l6 l25) (road l6 l37) (road l7 l35) (road l7 l49) (road l8 l2) (road l8 l23) (road l8 l39) (road l9 l24) (road l9 l36) (spare-in l12) (spare-in l13) (spare-in l14) (spare-in l16) (spare-in l2) (spare-in l20) (spare-in l22) (spare-in l23) (spare-in l25) (spare-in l26) (spare-in l28) (spare-in l29) (spare-in l3) (spare-in l32) (spare-in l34) (spare-in l35) (spare-in l36) (spare-in l40) (spare-in l41) (spare-in l42) (spare-in l44) (spare-in l46) (spare-in l48) (spare-in l51) (spare-in l52) (spare-in l6) (spare-in l7) (spare-in l8) (vehicle-at l50))
    (:goal (vehicle-at l30))
)