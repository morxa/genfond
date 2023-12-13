(define (problem tireworld-080-02)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l22 l23 l24 l25 l26 l27 l28 l29 l3 l30 l31 l32 l33 l34 l35 l36 l37 l38 l39 l4 l40 l41 l42 l43 l44 l45 l46 l47 l48 l49 l5 l50 l51 l52 l53 l54 l55 l56 l57 l58 l59 l6 l60 l61 l62 l63 l64 l65 l66 l67 l68 l69 l7 l70 l71 l72 l73 l74 l75 l76 l77 l78 l79 l8 l9 - location)
    (:init (road l0 l70) (road l0 l76) (road l1 l39) (road l1 l73) (road l10 l31) (road l10 l51) (road l10 l60) (road l10 l64) (road l11 l74) (road l11 l75) (road l12 l25) (road l12 l46) (road l13 l42) (road l13 l50) (road l14 l27) (road l14 l46) (road l15 l72) (road l15 l77) (road l16 l37) (road l16 l7) (road l17 l40) (road l17 l59) (road l18 l49) (road l18 l57) (road l19 l39) (road l19 l66) (road l2 l53) (road l20 l71) (road l20 l73) (road l21 l37) (road l21 l61) (road l22 l39) (road l22 l43) (road l23 l41) (road l23 l44) (road l24 l4) (road l24 l69) (road l25 l12) (road l25 l54) (road l26 l30) (road l26 l47) (road l26 l75) (road l27 l14) (road l27 l6) (road l28 l51) (road l28 l77) (road l29 l32) (road l29 l57) (road l3 l35) (road l3 l39) (road l30 l26) (road l30 l34) (road l31 l10) (road l31 l43) (road l32 l29) (road l32 l62) (road l33 l58) (road l33 l9) (road l34 l30) (road l34 l48) (road l34 l64) (road l35 l3) (road l35 l38) (road l36 l43) (road l36 l50) (road l37 l16) (road l37 l21) (road l37 l76) (road l38 l35) (road l38 l69) (road l39 l1) (road l39 l19) (road l39 l22) (road l39 l3) (road l4 l24) (road l4 l6) (road l40 l17) (road l40 l63) (road l41 l23) (road l41 l55) (road l42 l13) (road l42 l7) (road l43 l22) (road l43 l31) (road l43 l36) (road l43 l48) (road l43 l5) (road l44 l23) (road l44 l53) (road l45 l55) (road l45 l67) (road l46 l12) (road l46 l14) (road l47 l26) (road l47 l9) (road l48 l34) (road l48 l43) (road l49 l18) (road l49 l58) (road l5 l43) (road l5 l72) (road l50 l13) (road l50 l36) (road l51 l10) (road l51 l28) (road l52 l54) (road l52 l61) (road l53 l2) (road l53 l44) (road l54 l25) (road l54 l52) (road l55 l41) (road l55 l45) (road l56 l70) (road l56 l75) (road l57 l18) (road l57 l29) (road l58 l33) (road l58 l49) (road l59 l17) (road l59 l78) (road l6 l27) (road l6 l4) (road l60 l10) (road l60 l70) (road l61 l21) (road l61 l52) (road l62 l32) (road l62 l8) (road l63 l40) (road l63 l66) (road l64 l10) (road l64 l34) (road l65 l68) (road l65 l72) (road l66 l19) (road l66 l63) (road l67 l45) (road l67 l78) (road l68 l65) (road l68 l73) (road l69 l24) (road l69 l38) (road l7 l16) (road l7 l42) (road l70 l0) (road l70 l56) (road l70 l60) (road l71 l20) (road l71 l8) (road l72 l15) (road l72 l5) (road l72 l65) (road l73 l1) (road l73 l20) (road l73 l68) (road l74 l11) (road l74 l79) (road l75 l11) (road l75 l26) (road l75 l56) (road l76 l0) (road l76 l37) (road l77 l15) (road l77 l28) (road l78 l59) (road l78 l67) (road l79 l74) (road l8 l62) (road l8 l71) (road l9 l33) (road l9 l47) (spare-in l0) (spare-in l10) (spare-in l11) (spare-in l12) (spare-in l14) (spare-in l15) (spare-in l16) (spare-in l17) (spare-in l18) (spare-in l19) (spare-in l20) (spare-in l21) (spare-in l22) (spare-in l23) (spare-in l24) (spare-in l25) (spare-in l26) (spare-in l27) (spare-in l28) (spare-in l3) (spare-in l30) (spare-in l31) (spare-in l32) (spare-in l33) (spare-in l34) (spare-in l35) (spare-in l36) (spare-in l37) (spare-in l38) (spare-in l39) (spare-in l4) (spare-in l40) (spare-in l41) (spare-in l42) (spare-in l43) (spare-in l44) (spare-in l45) (spare-in l46) (spare-in l48) (spare-in l49) (spare-in l5) (spare-in l50) (spare-in l51) (spare-in l52) (spare-in l53) (spare-in l54) (spare-in l55) (spare-in l56) (spare-in l57) (spare-in l58) (spare-in l59) (spare-in l6) (spare-in l61) (spare-in l62) (spare-in l63) (spare-in l64) (spare-in l65) (spare-in l66) (spare-in l67) (spare-in l68) (spare-in l69) (spare-in l7) (spare-in l70) (spare-in l71) (spare-in l72) (spare-in l73) (spare-in l74) (spare-in l75) (spare-in l76) (spare-in l77) (spare-in l78) (spare-in l8) (spare-in l9) (vehicle-at l79))
    (:goal (vehicle-at l2))
)