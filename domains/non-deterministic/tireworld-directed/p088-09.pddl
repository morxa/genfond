(define (problem tireworld-088-09)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l22 l23 l24 l25 l26 l27 l28 l29 l3 l30 l31 l32 l33 l34 l35 l36 l37 l38 l39 l4 l40 l41 l42 l43 l44 l45 l46 l47 l48 l49 l5 l50 l51 l52 l53 l54 l55 l56 l57 l58 l59 l6 l60 l61 l62 l63 l64 l65 l66 l67 l68 l69 l7 l70 l71 l72 l73 l74 l75 l76 l77 l78 l79 l8 l80 l81 l82 l83 l84 l85 l86 l87 l9 - location)
    (:init (road l0 l21) (road l0 l86) (road l1 l41) (road l1 l84) (road l10 l14) (road l10 l35) (road l11 l66) (road l11 l83) (road l12 l27) (road l12 l46) (road l13 l3) (road l13 l84) (road l14 l10) (road l14 l20) (road l15 l22) (road l15 l49) (road l16 l28) (road l16 l82) (road l17 l34) (road l17 l6) (road l18 l4) (road l18 l8) (road l19 l62) (road l19 l78) (road l2 l74) (road l2 l85) (road l20 l14) (road l20 l61) (road l21 l0) (road l21 l39) (road l22 l15) (road l22 l29) (road l23 l60) (road l23 l65) (road l24 l57) (road l24 l63) (road l25 l58) (road l25 l78) (road l26 l36) (road l26 l76) (road l27 l12) (road l27 l45) (road l28 l16) (road l28 l84) (road l29 l22) (road l29 l52) (road l3 l13) (road l3 l31) (road l30 l56) (road l30 l69) (road l31 l3) (road l31 l72) (road l31 l9) (road l32 l54) (road l32 l64) (road l33 l58) (road l33 l81) (road l34 l17) (road l34 l44) (road l35 l10) (road l35 l64) (road l36 l26) (road l36 l44) (road l37 l40) (road l37 l82) (road l38 l54) (road l38 l6) (road l39 l21) (road l39 l61) (road l4 l18) (road l4 l47) (road l4 l68) (road l40 l37) (road l40 l87) (road l41 l1) (road l41 l43) (road l42 l67) (road l43 l41) (road l43 l57) (road l44 l34) (road l44 l36) (road l45 l27) (road l45 l87) (road l46 l12) (road l46 l73) (road l47 l4) (road l47 l62) (road l48 l69) (road l48 l75) (road l49 l15) (road l49 l84) (road l5 l55) (road l5 l85) (road l50 l59) (road l50 l77) (road l51 l7) (road l51 l80) (road l52 l29) (road l52 l70) (road l53 l80) (road l53 l86) (road l54 l32) (road l54 l38) (road l55 l5) (road l55 l59) (road l56 l30) (road l56 l71) (road l57 l24) (road l57 l43) (road l57 l77) (road l58 l25) (road l58 l33) (road l59 l50) (road l59 l55) (road l59 l60) (road l6 l17) (road l6 l38) (road l60 l23) (road l60 l59) (road l61 l20) (road l61 l39) (road l62 l19) (road l62 l47) (road l63 l24) (road l63 l8) (road l64 l32) (road l64 l35) (road l65 l23) (road l65 l70) (road l66 l11) (road l66 l79) (road l67 l42) (road l67 l76) (road l68 l4) (road l68 l73) (road l69 l30) (road l69 l48) (road l7 l51) (road l7 l81) (road l70 l52) (road l70 l65) (road l71 l56) (road l71 l74) (road l72 l31) (road l72 l83) (road l73 l46) (road l73 l68) (road l74 l2) (road l74 l71) (road l75 l48) (road l75 l9) (road l76 l26) (road l76 l67) (road l77 l50) (road l77 l57) (road l78 l19) (road l78 l25) (road l79 l66) (road l8 l18) (road l8 l63) (road l80 l51) (road l80 l53) (road l81 l33) (road l81 l7) (road l82 l16) (road l82 l37) (road l83 l11) (road l83 l72) (road l84 l1) (road l84 l13) (road l84 l28) (road l84 l49) (road l85 l2) (road l85 l5) (road l86 l0) (road l86 l53) (road l87 l45) (road l9 l31) (road l9 l75) (spare-in l0) (spare-in l1) (spare-in l10) (spare-in l11) (spare-in l12) (spare-in l13) (spare-in l14) (spare-in l15) (spare-in l16) (spare-in l17) (spare-in l19) (spare-in l2) (spare-in l20) (spare-in l21) (spare-in l22) (spare-in l23) (spare-in l25) (spare-in l26) (spare-in l27) (spare-in l28) (spare-in l29) (spare-in l30) (spare-in l31) (spare-in l32) (spare-in l33) (spare-in l34) (spare-in l35) (spare-in l36) (spare-in l37) (spare-in l38) (spare-in l39) (spare-in l4) (spare-in l40) (spare-in l43) (spare-in l44) (spare-in l45) (spare-in l46) (spare-in l47) (spare-in l48) (spare-in l49) (spare-in l5) (spare-in l50) (spare-in l51) (spare-in l52) (spare-in l53) (spare-in l54) (spare-in l55) (spare-in l56) (spare-in l57) (spare-in l58) (spare-in l59) (spare-in l6) (spare-in l60) (spare-in l61) (spare-in l62) (spare-in l64) (spare-in l65) (spare-in l66) (spare-in l67) (spare-in l68) (spare-in l69) (spare-in l7) (spare-in l70) (spare-in l71) (spare-in l72) (spare-in l73) (spare-in l74) (spare-in l75) (spare-in l76) (spare-in l77) (spare-in l78) (spare-in l80) (spare-in l81) (spare-in l82) (spare-in l83) (spare-in l84) (spare-in l85) (spare-in l86) (spare-in l87) (spare-in l9) (vehicle-at l79))
    (:goal (vehicle-at l42))
)