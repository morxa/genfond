(define (problem tireworld-078-09)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l22 l23 l24 l25 l26 l27 l28 l29 l3 l30 l31 l32 l33 l34 l35 l36 l37 l38 l39 l4 l40 l41 l42 l43 l44 l45 l46 l47 l48 l49 l5 l50 l51 l52 l53 l54 l55 l56 l57 l58 l59 l6 l60 l61 l62 l63 l64 l65 l66 l67 l68 l69 l7 l70 l71 l72 l73 l74 l75 l76 l77 l8 l9 - location)
    (:init (road l0 l16) (road l0 l39) (road l1 l46) (road l1 l67) (road l10 l19) (road l10 l64) (road l10 l74) (road l11 l20) (road l11 l28) (road l12 l24) (road l12 l30) (road l13 l14) (road l13 l5) (road l14 l13) (road l14 l63) (road l15 l26) (road l15 l31) (road l15 l44) (road l16 l0) (road l16 l49) (road l17 l25) (road l17 l43) (road l18 l46) (road l18 l73) (road l19 l10) (road l19 l70) (road l2 l39) (road l2 l67) (road l20 l11) (road l20 l56) (road l21 l61) (road l21 l65) (road l22 l51) (road l22 l58) (road l23 l41) (road l23 l43) (road l24 l12) (road l24 l65) (road l25 l17) (road l25 l63) (road l26 l15) (road l26 l45) (road l27 l35) (road l27 l60) (road l28 l11) (road l28 l30) (road l29 l40) (road l29 l48) (road l29 l69) (road l3 l37) (road l3 l9) (road l30 l12) (road l30 l28) (road l31 l15) (road l31 l42) (road l32 l34) (road l32 l56) (road l32 l70) (road l33 l59) (road l33 l64) (road l34 l32) (road l34 l68) (road l34 l69) (road l35 l27) (road l35 l7) (road l36 l37) (road l36 l74) (road l37 l3) (road l37 l36) (road l38 l52) (road l39 l0) (road l39 l2) (road l4 l72) (road l4 l9) (road l40 l29) (road l40 l52) (road l41 l23) (road l41 l73) (road l42 l31) (road l42 l52) (road l43 l17) (road l43 l23) (road l44 l15) (road l44 l66) (road l45 l26) (road l45 l75) (road l46 l1) (road l46 l18) (road l47 l64) (road l47 l66) (road l48 l29) (road l48 l62) (road l49 l16) (road l49 l57) (road l5 l13) (road l5 l6) (road l50 l72) (road l50 l77) (road l51 l22) (road l51 l77) (road l52 l38) (road l52 l40) (road l52 l42) (road l53 l55) (road l53 l62) (road l54 l55) (road l54 l71) (road l55 l53) (road l55 l54) (road l56 l20) (road l56 l32) (road l56 l73) (road l57 l49) (road l57 l58) (road l58 l22) (road l58 l57) (road l59 l33) (road l59 l76) (road l6 l5) (road l6 l8) (road l60 l27) (road l61 l21) (road l61 l76) (road l62 l48) (road l62 l53) (road l63 l14) (road l63 l25) (road l64 l10) (road l64 l33) (road l64 l47) (road l65 l21) (road l65 l24) (road l66 l44) (road l66 l47) (road l67 l1) (road l67 l2) (road l68 l34) (road l68 l75) (road l69 l29) (road l69 l34) (road l7 l35) (road l7 l70) (road l70 l19) (road l70 l32) (road l70 l7) (road l71 l54) (road l71 l8) (road l72 l4) (road l72 l50) (road l73 l18) (road l73 l41) (road l73 l56) (road l74 l10) (road l74 l36) (road l75 l45) (road l75 l68) (road l76 l59) (road l76 l61) (road l77 l50) (road l77 l51) (road l8 l6) (road l8 l71) (road l9 l3) (road l9 l4) (spare-in l13) (spare-in l15) (spare-in l26) (spare-in l27) (spare-in l31) (spare-in l32) (spare-in l34) (spare-in l35) (spare-in l42) (spare-in l43) (spare-in l44) (spare-in l45) (spare-in l52) (spare-in l63) (spare-in l68) (spare-in l7) (spare-in l70) (spare-in l75) (vehicle-at l60))
    (:goal (vehicle-at l38))
)